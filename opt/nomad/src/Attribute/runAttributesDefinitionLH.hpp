//////////// THIS FILE MUST BE CREATED BY EXECUTING WriteAttributeDefinitionFile ////////////
//////////// DO NOT MODIFY THIS FILE MANUALLY ///////////////////////////////////////////////

#ifndef __NOMAD_4_0_RUNATTRIBUTESDEFINITIONLH__
#define __NOMAD_4_0_RUNATTRIBUTESDEFINITIONLH__

_definition = {
{ "LH_EVAL",  "size_t",  "0",  " Latin Hypercube Sampling of points (no optimization) ",  " \n  \n . Latin-Hypercube sampling (evaluations) \n  \n . Argument: A positive integer p < INF \n  \n . p: number of LH points \n  \n . All points will be evaluated (no opportunism). This options will deactivate \n   any optimization strategy. \n  \n . The LH sampling requires to have both lower and upper bounds defined. \n  \n . Example: LH_EVAL 100 \n  \n . Default: 0\n\n",  "  basic latin hypercube sampling  "  , "true" , "true" , "true" },
{ "LH_SEARCH",  "NOMAD::LHSearchType",  "-",  " Latin Hypercube Sampling Search method ",  " \n  \n . Latin-Hypercube sampling (search) \n  \n . Arguments: two size_t p0 and pi \n  \n . p0: number of initial LH search points \n  \n . pi: LH search points at each iteration \n  \n . The search can be opportunistic or not \n   (parameter EVAL_OPPORTUNISTIC) \n  \n . Example: LH_SEARCH 100 0 \n  \n . No default value.\n\n",  "  basic search latin hypercube sampling  "  , "true" , "true" , "true" } };

#endif
