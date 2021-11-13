//////////// THIS FILE MUST BE CREATED BY EXECUTING WriteAttributeDefinitionFile ////////////
//////////// DO NOT MODIFY THIS FILE MANUALLY ///////////////////////////////////////////////

#ifndef __NOMAD_4_0_RUNATTRIBUTESDEFINITIONVNS__
#define __NOMAD_4_0_RUNATTRIBUTESDEFINITIONVNS__

_definition = {
{ "VNS_MADS_OPTIMIZATION",  "bool",  "false",  " VNS MADS stand alone optimization for constrained and unconstrained pbs ",  " \n  \n . Shaking + optimization for constrained and unconstrained optimization \n  \n . Argument: bool \n  \n . Stand alone VNS Mads optimization will deactivate any optimization strategy. \n  \n . Example: VNS_MADS_OPTIMIZATION true \n  \n . Default: false\n\n",  "  advanced global optimization vns neighborhood  "  , "true" , "false" , "true" },
{ "VNS_MADS_SEARCH",  "bool",  "false",  " VNS Mads optimization used as a search step for Mads ",  " \n  \n . Variable Neighborhood Search + Mads optimization as a search step for Mads \n  \n . Argument: bool \n  \n . Example: VNS_MADS_SEARCH false \n  \n . Default: false\n\n",  "  advanced global mads search vns neighborhood "  , "true" , "true" , "true" },
{ "VNS_MADS_SEARCH_TRIGGER",  "NOMAD::Double",  "0.75",  " VNS Mads search trigger",  " \n  \n . The VNS trigger is the maximum desired ratio of VNS blackbox evaluations \n   over the total number of blackbox evaluations. \n    \n . The VNS search is never executed with a null trigger while a value of 1 \n   allows the search at every iteration \n    \n . If \"VNS_MADS_SEARCH yes\", the default value of 0.75 is taken for the trigger \n  \n . Argument: Double \n  \n . Example: VNS_MADS_SEARCH_TRIGGER 0.9 \n  \n . Default: 0.75\n\n",  "  advanced global mads search vns neighborhood ratio  "  , "true" , "true" , "true" },
{ "VNS_MADS_SEARCH_MAX_TRIAL_PTS_NFACTOR",  "size_t",  "100",  " VNS-Mads search stopping criterion.",  " \n  \n . VNS Mads stopping criterion. Max number of trial pts < dimension * NFactor \n  \n . Argument: Positive integer. INF disables this criterion. \n  \n . Example: VNS_MADS_SEARCH_MAX_TRIAL_PTS_NFACTOR 10 \n  \n . Default: 100\n\n",  "  advanced global vns neighborhood mads search stop trial  "  , "true" , "true" , "true" } };

#endif
