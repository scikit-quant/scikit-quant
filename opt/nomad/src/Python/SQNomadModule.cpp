// Copyright UC Regents

#include "PyCompat.h"
#include "../nomad_platform.hpp"
#include "../Nomad/nomad.hpp"
#include "../Cache/CacheBase.hpp"
#include "../Type/BBOutputType.hpp"
#include "../Algos/EvcInterface.hpp"
#include "../Math/RNG.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"     // numpy


using namespace NOMAD;  // versioned


//- data ---------------------------------------------------------------------
namespace SQNomad {
    PyObject* gThisModule    = nullptr;
}


//- helpers ------------------------------------------------------------------
namespace {

class PyGILRAII {
    PyGILState_STATE m_GILState;
public:
    PyGILRAII() : m_GILState(PyGILState_Ensure()) {}
    ~PyGILRAII() { PyGILState_Release(m_GILState); }
};

class PyUnGILRAII {
    PyThreadState* m_ThreadState;
public:
    PyUnGILRAII() { m_ThreadState = PyEval_SaveThread(); }
    ~PyUnGILRAII() { PyEval_RestoreThread(m_ThreadState); }
};

class EvcInterfaceReset {
    MainStep& m_cnomad;

public:
    EvcInterfaceReset(MainStep& m) : m_cnomad(m) {}
    ~EvcInterfaceReset() {
    // resetting the optimizer state is necessary to allow re-runs within the
    // same program; this also takes care of resetting the evaluator in case
    // the user level library gets unloaded earlier than NOMAD
        m_cnomad.resetComponentsBetweenOptimization();
    }
};

} // unnamed namespace


//----------------------------------------------------------------------------
static Py_ssize_t GetBuffer(PyObject* pyobject, char tc, int size, void*& buf, bool check)
{
// Retrieve a linear buffer pointer from the given pyobject.

// special case: don't handle character strings here (yes, they're buffers, but not quite)
    if (PyBytes_Check(pyobject) || PyUnicode_Check(pyobject))
        return 0;

// special case: bytes array
    if ((!check || tc == '*' || tc == 'B') && PyByteArray_CheckExact(pyobject)) {
        buf = PyByteArray_AS_STRING(pyobject);
        return PyByteArray_GET_SIZE(pyobject);
    }

// new-style buffer interface
    if (PyObject_CheckBuffer(pyobject)) {
        Py_buffer bufinfo;
        memset(&bufinfo, 0, sizeof(Py_buffer));
        if (PyObject_GetBuffer(pyobject, &bufinfo, PyBUF_FORMAT) == 0) {
            if (tc == '*' || strchr(bufinfo.format, tc)
#ifdef _WIN32
            // ctypes is inconsistent in format on Windows; either way these types are the same size
                || (tc == 'I' && strchr(bufinfo.format, 'L')) || (tc == 'i' && strchr(bufinfo.format, 'l'))
#endif
            // complex float is 'Zf' in bufinfo.format, but 'z' in single char
                || (tc == 'z' && strstr(bufinfo.format, "Zf"))
            // allow 'signed char' ('b') from array to pass through '?' (bool as from struct)
                || (tc == '?' && strchr(bufinfo.format, 'b'))
                    ) {
                buf = bufinfo.buf;
                Py_ssize_t buflen = 0;
                if (buf && bufinfo.ndim == 0)
                    buflen = bufinfo.len/bufinfo.itemsize;
                else if (buf && bufinfo.ndim == 1)
                    buflen = bufinfo.shape ? bufinfo.shape[0] : bufinfo.len/bufinfo.itemsize;
                PyCompat_PyBuffer_Release(pyobject, &bufinfo);
                if (buflen)
                    return buflen;
            } else {
            // have buf, but format mismatch: bail out now, otherwise the old
            // code will return based on itemsize match
                PyCompat_PyBuffer_Release(pyobject, &bufinfo);
                return 0;
            }
        } else if (bufinfo.obj)
            PyCompat_PyBuffer_Release(pyobject, &bufinfo);
        PyErr_Clear();
    }

// attempt to retrieve pointer through old-style buffer interface
    PyBufferProcs* bufprocs = Py_TYPE(pyobject)->tp_as_buffer;

    PySequenceMethods* seqmeths = Py_TYPE(pyobject)->tp_as_sequence;
    if (seqmeths != 0 && bufprocs != 0
#if PY_VERSION_HEX < 0x03000000
         && bufprocs->bf_getwritebuffer != 0
         && (*(bufprocs->bf_getsegcount))(pyobject, 0) == 1
#else
         && bufprocs->bf_getbuffer != 0
#endif
        ) {

   // get the buffer
#if PY_VERSION_HEX < 0x03000000
        Py_ssize_t buflen = (*(bufprocs->bf_getwritebuffer))(pyobject, 0, &buf);
#else
        Py_buffer bufinfo;
        (*(bufprocs->bf_getbuffer))(pyobject, &bufinfo, PyBUF_WRITABLE);
        buf = (char*)bufinfo.buf;
        Py_ssize_t buflen = bufinfo.len;
        PyCompat_PyBuffer_Release(pyobject, &bufinfo);
#endif

        if (buf && check == true) {
        // determine buffer compatibility (use "buf" as a status flag)
            PyObject* pytc = PyObject_GetAttrString(pyobject, (char*)"typecode");
            if (pytc != 0) {      // for array objects
                char cpytc = PyCompat_PyText_AsString(pytc)[0];
                if (!(cpytc == tc || (tc == '?' && cpytc == 'b')))
                    buf = 0;      // no match
                Py_DECREF(pytc);
            } else if (seqmeths->sq_length &&
                       (int)(buflen/(*(seqmeths->sq_length))(pyobject)) == size) {
            // this is a gamble ... may or may not be ok, but that's for the user
                PyErr_Clear();
            } else if (buflen == size) {
            // also a gamble, but at least 1 item will fit into the buffer, so very likely ok ...
                PyErr_Clear();
            } else {
                buf = 0;                      // not compatible

            // clarify error message
                PyObject* pytype = 0, *pyvalue = 0, *pytrace = 0;
                PyErr_Fetch(&pytype, &pyvalue, &pytrace);
                PyObject* pyvalue2 = PyCompat_PyText_FromFormat(
                    (char*)"%s and given element size (%ld) do not match needed (%d)",
                    PyCompat_PyText_AsString(pyvalue),
                    seqmeths->sq_length ? (long)(buflen/(*(seqmeths->sq_length))(pyobject)) : (long)buflen,
                    size);
                Py_DECREF(pyvalue);
                PyErr_Restore(pytype, pyvalue2, pytrace);
            }
        }

        if (!buf) return 0;
        return buflen/(size ? size : 1);
    }

    return 0;
}


//----------------------------------------------------------------------------
class PyCallback : public Evaluator {
private:
    PyObject* m_callback;

public:
    PyCallback(std::shared_ptr<EvalParameters> ep, PyObject* f) :
            Evaluator(ep) {
        Py_INCREF(f);
        m_callback = f;
    }
    PyCallback(const PyCallback& pc) : Evaluator(pc) {
        Py_INCREF(pc.m_callback);
        m_callback = pc.m_callback;
    }
    PyCallback(const PyCallback&& pc) : Evaluator(pc) {
        Py_INCREF(pc.m_callback);
        m_callback = pc.m_callback;
    }
    PyCallback& operator=(const PyCallback&) = delete;
    ~PyCallback() {
        PyGILRAII m;              // as Py_DECREF may cause deletion
        Py_DECREF(m_callback);
    }

public:
    std::vector<bool> eval_block(Block& block,
            const Double& NOMAD_UNUSED(hMax),
            std::vector<bool>& countEval) const override {

        const size_t np = block.size();
        std::vector<bool> evalOk(np, false);
        countEval.resize(np, false);

        auto bbOutputType = _evalParams->getAttributeValue<NOMAD::BBOutputTypeList>("BB_OUTPUT_TYPE");

     // TODO: this code can be called in parallel by NOMAD; need to test whether
     // to place GIL acquisition inside the loop itself where needed, rathern than
     // on the outside for the full calculation
        PyGILRAII m;

        PyObject *py_x = nullptr, *args = nullptr;
        for (size_t i = 0; i < np; ++i) {
            EvalPointPtr x_ptr = block[0];
            const Point* x = x_ptr->getX();
            if (!py_x) {
                npy_intp dims[] = {(npy_intp)x->size()};
                py_x = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
                args = PyTuple_New(1);
                PyTuple_SET_ITEM(args, 0, py_x);
            }
            double* buf = (double*)PyArray_DATA((PyArrayObject*)py_x);
            for (size_t j = 0; j < x->size(); ++j) buf[j] = (*x)[j].todouble();
            PyObject* res = PyObject_CallObject(m_callback, args);
            evalOk[i] = false;
            if (res) {
                std::ostringstream sbbo;

                if (PyTuple_Check(res)) {
                    evalOk[i] = true;
                    for (Py_ssize_t resi = 0; resi < PyTuple_GET_SIZE(res); ++resi) {
                        double dres = PyFloat_AsDouble(PyTuple_GET_ITEM(res, resi));
                        if (dres == -1. && PyErr_Occurred()) {
                            evalOk[i] = false;
                            break;
                        }
                        if (resi != 0)
                            sbbo << " ";
                        sbbo << dres;
                    }
                } else {
                    double dres = PyFloat_AsDouble(res);
                    if (!(dres == -1. && PyErr_Occurred())) {
                        evalOk[i] = true;
                        sbbo << dres;
                    }
                }

                if (evalOk[i]) {
                    const NOMAD::EvalType& evalType = getEvalType();
                    x_ptr->setBBO(sbbo.str(), bbOutputType, evalType);
                }

                Py_DECREF(res);
            }

            if (!evalOk[i])
                PyErr_Print();        // TODO: fetch errors and reset on return

            countEval[i] = true;
        }
        Py_XDECREF(args);

        return evalOk;
    }
};


//----------------------------------------------------------------------------
static PyObject* minimize(PyObject* /* dummy */, PyObject* args, PyObject* kwds)
{
// Convert Python objects into C++ structures for NOMAD and handle any C++ exceptions
// that may arise. In principle, this function is only called from the SQNomad Python
// module, but the main error checking is here, to allow re-use.

// required arguments are objective, initial, lower bounds, upper bounds
    if (PyTuple_GET_SIZE(args) != 4) {
        PyErr_Format(PyExc_TypeError,
            "minimize() takes 4 positional arguments but " PY_SSIZE_T_FORMAT " were given",
            PyTuple_GET_SIZE(args));
        return nullptr;
    }

    PyObject* result = nullptr;
    try {     // big outer try/except to prevent NOMAD raised exceptions getting to python

    // collect the parameters for NOMAD
        auto params = std::make_shared<AllParameters>();

    // check objective
        if (!PyCallable_Check(PyTuple_GET_ITEM(args, 0))) {
            PyErr_SetString(PyExc_TypeError, "objective function is not callable");
            return nullptr;
        }

    // get initial
        double* par = nullptr;
        Py_ssize_t npar = GetBuffer(PyTuple_GET_ITEM(args, 1), 'd', sizeof(double), (void*&)par, true);
        if (!par || npar == 0) {
            PyErr_SetString(PyExc_TypeError, "unable to convert parameter array");
            return nullptr;
        }

        Point x0(npar);
        for (Py_ssize_t i = 0; i < npar; ++i)
            x0[i] = par[i];
        params->setAttributeValue("X0", x0);

    // get the bounds buffers, if provided
        double *lower = nullptr, *upper = nullptr;
        for (Py_ssize_t iarg : {2, 3}) {
            if (PyTuple_GET_ITEM(args, iarg) == Py_None)
                continue;

            void*& bounds = (void*&)(iarg == 2 ? lower : upper);
            Py_ssize_t nbounds = GetBuffer(PyTuple_GET_ITEM(args, iarg), 'd', sizeof(double), bounds, true);
            if (!bounds || nbounds == 0) {
                PyErr_SetString(PyExc_TypeError, "unable to convert bounds array");
                return nullptr;
            }

            if (npar != nbounds) {
                PyErr_Format(PyExc_ValueError,
                    "length of initial (" PY_SSIZE_T_FORMAT ") and %s bounds (" PY_SSIZE_T_FORMAT ") do not match",
                    npar, (iarg == 2 ? "lower" : "upper"), nbounds);
                return nullptr;
            }
        }

        if (lower) {
            ArrayOfDouble lbounds(npar);
            for (Py_ssize_t i = 0; i < npar; ++i)
                lbounds[i] = lower[i];
            params->setAttributeValue("LOWER_BOUND", lbounds);
        }

        if (upper) {
            ArrayOfDouble ubounds(npar);
            for (Py_ssize_t i = 0; i < npar; ++i) {
                if (lower && upper[i] <= lower[i]) {
                    PyErr_Format(PyExc_ValueError,
                        "for element " PY_SSIZE_T_FORMAT ", upper bound (%d) is smaller than lower bound (%d)",
                        i, upper[i], lower[i]);
                    return nullptr;
                }
                ubounds[i] = upper[i];
            }
            params->setAttributeValue("UPPER_BOUND", ubounds);
        }

    // process known options (allow unknown options to pass as strings)
        std::vector<bool> options_ok{
            true,  /* all okay */
            false, /* have BB_OUTPUT_TYPE */
            false, /* have DISPLAY_DEGREE */
            false, /* have DISPLAY_ALL_EVAL */
            false, /* have DIMENSION */
            false  /* have SEED */ };

        if (kwds) {
            PyObject* items = PyDict_Items(kwds);
            for (Py_ssize_t i = 0; i < PyList_GET_SIZE(items); ++i) {
                PyObject* pair = PyList_GET_ITEM(items, i);

                const char* key = PyCompat_PyText_AsString(PyTuple_GET_ITEM(pair, 0));
                if (!key) {
                    options_ok[0] = false;
                    break;
                }

                if (strcmp(key, "MAX_BB_EVAL") == 0) {
                    size_t budget = PyLong_AsSize_t(PyTuple_GET_ITEM(pair, 1));
                    if (budget == (size_t)-1 && PyErr_Occurred()) {
                        options_ok[0] = false;
                        break;
                    }
                    params->setAttributeValue<size_t>(key, (size_t)budget);

                // have to set MAX_EVAL along side MAX_BB_EVAL, unless overridden in options
                    PyObject* v = PyDict_GetItemString(kwds, "MAX_EVAL");
                    if (!v) {
                        PyErr_Clear();
                        params->setAttributeValue<size_t>("MAX_EVAL", (size_t)100*budget);
                    }

                } else if (strcmp(key, "MAX_EVAL") == 0) {
                    size_t mxeval = PyLong_AsSize_t(PyTuple_GET_ITEM(pair, 1));
                    if (mxeval == (size_t)-1 && PyErr_Occurred()) {
                        options_ok[0] = false;
                        break;
                    }
                    params->setAttributeValue<size_t>(key, (size_t)mxeval);

                } else if (strcmp(key, "BB_OUTPUT_TYPE") == 0) {
                    const char* value = PyCompat_PyText_AsString(PyTuple_GET_ITEM(pair, 1));
                    if (!value) {
                        PyErr_Format(PyExc_TypeError, "BB_OUTPUT_TYPE value field should be string");
                        options_ok[0] = false;
                        break;
                    }
                    options_ok[1] = true;
                    params->setAttributeValue(key, stringToBBOutputTypeList(value));

                } else if (strcmp(key, "SEED") == 0) {
                    const char* value = PyCompat_PyText_AsString(PyTuple_GET_ITEM(pair, 1));
                    if (!value) {
                        PyErr_Clear();
                        long seed = PyInt_AsLong(PyTuple_GET_ITEM(pair, 1));
                        if (seed == -1 && PyErr_Occurred()) {
                             options_ok[0] = false;
                             break;
                        }
                        params->setAttributeValue<int>(key, (int)seed);
                    } else
                        params->readParamLine(std::string(key) + " " + value);
                    options_ok[5] = true;

                } else {
                    if (strcmp(key, "BB_OUTPUT_TYPE") == 0)        options_ok[1] = true;
                    else if (strcmp(key, "DISPLAY_DEGREE") == 0)   options_ok[2] = true;
                    else if (strcmp(key, "DISPLAY_ALL_EVAL") == 0) options_ok[3] = true;
                    else if (strcmp(key, "DIMENSION") == 0)        options_ok[4] = true;

                    const char* value = PyCompat_PyText_AsString(PyTuple_GET_ITEM(pair, 1));
                    if (!value) {
                        PyErr_Format(PyExc_TypeError, "string expected for value of \'%s\'", key);
                        options_ok[0] = false;
                        break;
                    }
                    params->readParamLine(std::string(key) + " " + value);    // may fail on check later
                }
            }
            Py_DECREF(items);
        }

        if (!options_ok[0])
            return nullptr;  // error already set

    // add defaults/implied parameters
        if (!options_ok[1])
            params->setAttributeValue("BB_OUTPUT_TYPE", stringToBBOutputTypeList("OBJ"));
        if (!options_ok[2]) params->setAttributeValue("DISPLAY_DEGREE", 0);
        if (!options_ok[3]) params->setAttributeValue("DISPLAY_ALL_EVAL", false);
        if (!options_ok[4]) params->setAttributeValue("DIMENSION", (size_t)npar);

        params->getPbParams()->setAttributeValue("GRANULARITY", NOMAD::ArrayOfDouble(npar, 0.0000001));

        //if (!options_ok[5])
            RNG::resetPrivateSeedToDefault();

    // verify (should never fail at this point)
        params->checkAndComply();

    // reset the cache (TODO: let the user control this behavior, but w/o knowing what filled
    // the cache, using it implicitly is too error prone)
        try {
            const std::unique_ptr<CacheBase>& cache = CacheBase::getInstance();
            cache->clear();
        } catch (...) {
        // non-existent / nothing to clear
        }

    // create and configure NOMAD instance (outside of thread blocks b/c it will own a Python object)
        MainStep cnomad;
        EvcInterfaceReset e{cnomad};

        cnomad.setAllParameters(params);
        cnomad.setEvaluator(
            std::make_unique<PyCallback>(params->getEvalParams(), PyTuple_GET_ITEM(args, 0)));

        bool run_ok = false;
        {
        PyUnGILRAII m;
        cnomad.start();
        run_ok = cnomad.run();
        cnomad.end();
        }

        if (run_ok) {
            std::vector<EvalPoint> evpl;
            auto nf = CacheBase::getInstance()->findBestFeas(
                evpl, Point(), EvalType::BB, NOMAD::ComputeType::STANDARD, nullptr);
            if (0 < nf) {
                EvalPoint evres = evpl[0];
                result = PyTuple_New(2);

            // fbest
                PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(evres.getF().todouble()));

            // xbest
                const Point* x = evres.getX();
                npy_intp dims[] = {(npy_intp)x->size()};
                PyObject* py_x = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
                double* buf = (double*)PyArray_DATA((PyArrayObject*)py_x);
                for (size_t j = 0; j < x->size(); ++j) buf[j] = (*x)[j].todouble();
                PyTuple_SET_ITEM(result, 1, py_x);

            } else
                PyErr_SetString(PyExc_RuntimeError, "no feasible point");
        }

    } catch (const NOMAD::Exception& e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.what());
        return nullptr;
    } catch (const std::exception& e) {
        PyErr_Format(PyExc_RuntimeError, "[C++] %s", e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception");
        return nullptr;
    }

    if (!result && !PyErr_Occurred())
        PyErr_SetString(PyExc_RuntimeError, "NOMAD run failed");
    return result;
}


static PyMethodDef gSQNomadMethods[] = {
    {(char*) "minimize", (PyCFunction)minimize,
      METH_VARARGS | METH_KEYWORDS, (char*)"NOMAD minimizer"},
    {nullptr, nullptr, 0, nullptr}
};


#if PY_VERSION_HEX >= 0x03000000
struct module_state {
    PyObject* error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int sqnomadmodule_traverse(PyObject* m, visitproc visit, void* arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int sqnomadmodule_clear(PyObject* m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "libsqnomad",
    nullptr,
    sizeof(struct module_state),
    gSQNomadMethods,
    nullptr,
    sqnomadmodule_traverse,
    sqnomadmodule_clear,
    nullptr
};


//----------------------------------------------------------------------------
#define PYCOMPAT_INIT_ERROR return nullptr
extern "C" PyObject* PyInit_libsqnomad()
#else
#define PYCOMPAT_INIT_ERROR return
extern "C" void initlibsqnomad()
#endif
{
// Initialization of extension module libsqnomad.
    using namespace SQNomad;

// setup interpreter
    if (!Py_IsInitialized()) {
    // this happens if Cling comes in first
#if PY_VERSION_HEX < 0x03020000
        PyEval_InitThreads();
#endif
        Py_Initialize();
#if PY_VERSION_HEX >= 0x03020000
#if PY_VERSION_HEX < 0x03090000
        PyEval_InitThreads();
#endif
#endif
    }

// force numpy
    import_array()
    PyObject* np = PyImport_AddModule((char*)"numpy");
    if (!np)
        return nullptr;       // error already set
    // np is borrowed, but we only need to C dlls, which are never unloaded

// setup this module
#if PY_VERSION_HEX >= 0x03000000
    gThisModule = PyModule_Create(&moduledef);
#else
    gThisModule = Py_InitModule(const_cast<char*>("libsqnomad"), gSQNomadMethods);
#endif
    if (!gThisModule)
        PYCOMPAT_INIT_ERROR;

// keep gThisModule, but do not increase its reference count even as it is borrowed,
// or a self-referencing cycle would be created

#if PY_VERSION_HEX >= 0x03000000
    Py_INCREF(gThisModule);
    return gThisModule;
#endif
}
