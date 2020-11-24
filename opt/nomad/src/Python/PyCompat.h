// Copyright UC Regents

#ifndef PYNOMAD_PYCOMPAT_H
#define PYNOMAD_PYCOMPAT_H

#ifdef _WIN32
// Disable warning C4275: non dll-interface class
#pragma warning (disable : 4275)
// Disable warning C4251: needs to have dll-interface to be used by clients
#pragma warning (disable : 4251)
// Disable warning C4800: 'int' : forcing value to bool
#pragma warning (disable : 4800)
// Avoid that pyconfig.h decides using a #pragma what library python library to use
//#define MS_NO_COREDLL 1
#endif

// to prevent problems with fpos_t and redefinition warnings
#if defined(linux)

#include <stdio.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _FILE_OFFSET_BITS
#undef _FILE_OFFSET_BITS
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#endif // linux

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <sys/types.h>

typedef Py_ssize_t dim_t;
typedef dim_t* dims_t;

// for 3.3 support
#if PY_VERSION_HEX < 0x03030000
   typedef PyDictEntry* (*dict_lookup_func)(PyDictObject*, PyObject*, long);
#else
#if PY_VERSION_HEX >= 0x03060000
   typedef Py_ssize_t (*dict_lookup_func)(
       PyDictObject*, PyObject*, Py_hash_t, PyObject***, Py_ssize_t*);
#else
   struct PyDictKeyEntry;
   typedef PyDictKeyEntry* (*dict_lookup_func)(PyDictObject*, PyObject*, Py_hash_t, PyObject***);
#define PyDictEntry PyDictKeyEntry
#endif
#endif

// for 3.0 support (backwards compatibility, really)
#if PY_VERSION_HEX < 0x03000000
#define PyBytes_Check                  PyString_Check
#define PyBytes_CheckExact             PyString_CheckExact
#define PyBytes_AS_STRING              PyString_AS_STRING
#define PyBytes_AsString               PyString_AsString
#define PyBytes_AsStringAndSize        PyString_AsStringAndSize
#define PyBytes_GET_SIZE               PyString_GET_SIZE
#define PyBytes_Size                   PyString_Size
#define PyBytes_FromFormat             PyString_FromFormat
#define PyBytes_FromString             PyString_FromString
#define PyBytes_FromStringAndSize      PyString_FromStringAndSize

#define PyBytes_Type    PyString_Type

#define PyCompat_PyText_Check                 PyString_Check
#define PyCompat_PyText_CheckExact            PyString_CheckExact
#define PyCompat_PyText_AsString              PyString_AS_STRING
#define PyCompat_PyText_AsStringChecked       PyString_AsString
#define PyCompat_PyText_GET_SIZE              PyString_GET_SIZE
#define PyCompat_PyText_GetSize               PyString_Size
#define PyCompat_PyText_FromFormat            PyString_FromFormat
#define PyCompat_PyText_FromString            PyString_FromString
#define PyCompat_PyText_InternFromString      PyString_InternFromString
#define PyCompat_PyText_Append                PyString_Concat
#define PyCompat_PyText_AppendAndDel          PyString_ConcatAndDel
#define PyCompat_PyText_FromStringAndSize     PyString_FromStringAndSize

static inline const char* PyCompat_PyText_AsStringAndSize(PyObject* pystr, Py_ssize_t* size)
{
    const char* cstr = PyCompat_PyText_AsStringChecked(pystr);
    if (cstr) *size = PyCompat_PyText_GetSize(pystr);
    return cstr;
}

#define PyCompat_PyText_Type PyString_Type

static inline PyObject* PyCompat_PyCapsule_New(
        void* cobj, const char* /* name */, void (*destr)(void*))
{
    return PyCObject_FromVoidPtr(cobj, destr);
}
#define PyCompat_PyCapsule_CheckExact    PyCObject_Check
static inline void* PyCompat_PyCapsule_GetPointer(PyObject* capsule, const char* /* name */)
{
    return (void*)PyCObject_AsVoidPtr(capsule);
}

#define PYCOMPAT__long__ "__long__"
#define PYCOMPAT__idiv__ "__idiv__"
#define PYCOMPAT__div__  "__div__"
#define PYCOMPAT__next__ "next"

typedef long Py_hash_t;

#endif  // ! 3.0

// for 3.0 support (backwards compatibility, really)
#if PY_VERSION_HEX >= 0x03000000
#define PyCompat_PyText_Check              PyUnicode_Check
#define PyCompat_PyText_CheckExact         PyUnicode_CheckExact
#define PyCompat_PyText_AsString           PyUnicode_AsUTF8
#define PyCompat_PyText_AsStringChecked    PyUnicode_AsUTF8
#define PyCompat_PyText_GetSize            PyUnicode_GetSize
#define PyCompat_PyText_GET_SIZE           PyUnicode_GET_SIZE
#define PyCompat_PyText_FromFormat         PyUnicode_FromFormat
#define PyCompat_PyText_FromString         PyUnicode_FromString
#define PyCompat_PyText_InternFromString   PyUnicode_InternFromString
#define PyCompat_PyText_Append             PyUnicode_Append
#define PyCompat_PyText_AppendAndDel       PyUnicode_AppendAndDel
#define PyCompat_PyText_FromStringAndSize  PyUnicode_FromStringAndSize

#if PY_VERSION_HEX >= 0x03030000
#define _PyCompat_PyText_AsStringAndSize   PyUnicode_AsUTF8AndSize
#else
#define _PyCompat_PyText_AsStringAndSize   PyUnicode_AsStringAndSize
#endif  // >= 3.3

static inline const char* PyCompat_PyText_AsStringAndSize(PyObject* pystr, Py_ssize_t* size)
{
    const char* cstr = _PyCompat_PyText_AsStringAndSize(pystr, size);
    if (!cstr && PyBytes_CheckExact(pystr)) {
        PyErr_Clear();
        PyBytes_AsStringAndSize(pystr, (char**)&cstr, size);
    }
    return cstr;
}

#define PyCompat_PyText_Type PyUnicode_Type

#define PyIntObject          PyLongObject
#define PyInt_Check          PyLong_Check
#define PyInt_AsLong         PyLong_AsLong
#define PyInt_AS_LONG        PyLong_AsLong
#define PyInt_AsSsize_t      PyLong_AsSsize_t
#define PyInt_CheckExact     PyLong_CheckExact
#define PyInt_FromLong       PyLong_FromLong
#define PyInt_FromSsize_t    PyLong_FromSsize_t

#define PyInt_Type      PyLong_Type

#define PyCompat_PyCapsule_New           PyCapsule_New
#define PyCompat_PyCapsule_CheckExact    PyCapsule_CheckExact
#define PyCompat_PyCapsule_GetPointer    PyCapsule_GetPointer

#define PyCOMPAT__long__ "__int__"
#define PyCOMPAT__idiv__ "__itruediv__"
#define PyCOMPAT__div__  "__truediv__"
#define PyCOMPAT__next__ "__next__"

#define Py_TPFLAGS_HAVE_RICHCOMPARE 0
#define Py_TPFLAGS_CHECKTYPES 0

#define PyClass_Check   PyType_Check

#define PyBuffer_Type   PyMemoryView_Type
#endif  // ! 3.0

#if PY_VERSION_HEX >= 0x03020000
#define PyCompat_PySliceCast   PyObject*
#define PyUnicode_GetSize      PyUnicode_GetLength
#else
#define PyCompat_PySliceCast   PySliceObject*
#endif  // >= 3.2

// feature of 3.0 not in 2.5 and earlier
#if PY_VERSION_HEX < 0x02060000
#define PyVarObject_HEAD_INIT(type, size)                                     \
    PyObject_HEAD_INIT(type) size,
#define Py_TYPE(ob)             (((PyObject*)(ob))->ob_type)
#endif

// API changes in 2.5 (int -> Py_ssize_t) and 3.2 (PyUnicodeObject -> PyObject)
#if PY_VERSION_HEX < 0x03020000
static inline Py_ssize_t PyCompat_PyUnicode_AsWideChar(PyObject* pyobj, wchar_t* w, Py_ssize_t size)
{
#if PY_VERSION_HEX < 0x02050000
     return (Py_ssize_t)PyUnicode_AsWideChar((PyUnicodeObject*)pyobj, w, (int)size);
#else
     return PyUnicode_AsWideChar((PyUnicodeObject*)pyobj, w, size);
#endif
}
#else
#define PyCompat_PyUnicode_AsWideChar PyUnicode_AsWideChar
#endif

// backwards compatibility, pre python 2.5
#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#define PyInt_AsSsize_t PyInt_AsLong
#define PyInt_FromSsize_t PyInt_FromLong
# define PY_SSIZE_T_FORMAT "%d"
# if !defined(PY_SSIZE_T_MIN)
#  define PY_SSIZE_T_MAX INT_MAX
#  define PY_SSIZE_T_MIN INT_MIN
# endif
#define ssizeobjargproc intobjargproc
#define lenfunc         inquiry
#define ssizeargfunc    intargfunc

#define PyIndex_Check(obj)                                                    \
    (PyInt_Check(obj) || PyLong_Check(obj))

inline Py_ssize_t PyNumber_AsSsize_t(PyObject* obj, PyObject*) {
    return (Py_ssize_t)PyLong_AsLong(obj);
}

#else
# ifdef R__MACOSX
#  if SIZEOF_SIZE_T == SIZEOF_INT
#    if defined(MAC_OS_X_VERSION_10_4)
#       define PY_SSIZE_T_FORMAT "%ld"
#    else
#       define PY_SSIZE_T_FORMAT "%d"
#    endif
#  elif SIZEOF_SIZE_T == SIZEOF_LONG
#    define PY_SSIZE_T_FORMAT "%ld"
#  endif
# else
#  define PY_SSIZE_T_FORMAT "%zd"
# endif
#endif

#if PY_VERSION_HEX < 0x02020000
#define PyBool_FromLong  PyInt_FromLong
#endif

#if PY_VERSION_HEX < 0x03000000
// the following should quiet Solaris
#ifdef Py_False
#undef Py_False
#define Py_False ((PyObject*)(void*)&_Py_ZeroStruct)
#endif

#ifdef Py_True
#undef Py_True
#define Py_True ((PyObject*)(void*)&_Py_TrueStruct)
#endif
#endif

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#ifndef Py_RETURN_TRUE
#define Py_RETURN_TRUE return Py_INCREF(Py_True), Py_True
#endif

#ifndef Py_RETURN_FALSE
#define Py_RETURN_FALSE return Py_INCREF(Py_False), Py_False
#endif

#if PY_VERSION_HEX >= 0x03000000 && PY_VERSION_HEX < 0x03010000
#define PyCompat_PyBuffer PyBuffer_Release
#else
inline void PyCompat_PyBuffer_Release(PyObject* /* unused */, Py_buffer* view) {
    PyBuffer_Release(view);
}
#endif

#endif // !PYNOMAD_PYCOMPAT_H
