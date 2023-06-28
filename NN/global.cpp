/**
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 */

/**
 * global flags and variables
 *
 * These variables are set during the command-line parsing and before
 * any real action, should only be read afterwards to allow parallel
 * processing.
 */

#include "global.hpp"

// toggle debug mode
bool flag_ref_mode = false;
// random number generators must be seeded
bool flag_random_seed = true;
