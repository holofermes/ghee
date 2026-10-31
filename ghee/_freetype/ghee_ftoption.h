/* FreeType's build options for ghee: FreeType's own defaults, less the
 * compressed font formats, which would need zlib. ghee takes .ttf and .otf
 * files. */
#include <freetype/config/ftoption.h>

#undef FT_CONFIG_OPTION_USE_ZLIB
#undef FT_CONFIG_OPTION_USE_LZW
