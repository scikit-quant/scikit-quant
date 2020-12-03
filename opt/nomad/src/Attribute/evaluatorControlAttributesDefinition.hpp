//////////// THIS FILE MUST BE CREATED BY EXECUTING WriteAttributeDefinitionFile ////////////
//////////// DO NOT MODIFY THIS FILE MANUALLY ///////////////////////////////////////////////

#ifndef __NOMAD400_EVALUATORCONTROLATTRIBUTESDEFINITION__
#define __NOMAD400_EVALUATORCONTROLATTRIBUTESDEFINITION__

_definition = {
{ "OPPORTUNISTIC_EVAL",  "bool",  "true",  " Opportunistic strategy: Terminate evaluations as soon as a success is found ",  " \n  \n . Opportunistic strategy: Terminate evaluations as soon as a success is found \n  \n . This parameter is the default value for other OPPORTUNISTIC parameters, \n    including Search steps \n  \n . This parameter is the value used for Poll step \n  \n . Argument: one boolean (yes or no) \n  \n . Type 'nomad -h opportunistic' to see advanced options \n  \n . Example: OPPORTUNISTIC_EVAL no  # complete evaluations \n  \n . Default: true\n\n",  "  advanced opportunistic oppor eval evals evaluation evaluations terminate list success successes  "  , "true" , "true" , "true" },
{ "USE_CACHE",  "bool",  "true",  " Use cache in algorithms ",  " \n . When this parameter is false, the Cache is not used at all. Points may be \n   re-evaluated. \n  \n . Recommended when DIMENSION is large and evaluations are not costly. \n  \n . Cache may be used for top algorithm, and disabled for a sub-algorithm. \n  \n . If CACHE_FILE is non-empty, cache file will still be read and written. \n  \n . Default: true\n\n",  "  advanced  "  , "true" , "false" , "true" },
{ "MAX_BB_EVAL_IN_SUBPROBLEM",  "size_t",  "INF",  " Max number of evaluations for each subproblem ",  " \n  \n . Used in the context of Sequential Space Decomposition (SSD) MADS \n   and Parallel Space Decomposition (PSD) MADS algorithms. \n  \n . Select the max number of evaluations in each Mads subproblem. \n  \n . Argument: size_t \n  \n . Example: MAX_BB_EVAL_IN_SUBPROBLEM 10 \n  \n . Default: INF\n\n",  "  advanced psd ssd mads parallel sequential decomposition subproblem  "  , "true" , "false" , "true" } };

#endif
