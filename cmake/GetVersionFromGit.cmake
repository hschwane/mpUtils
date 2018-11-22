include(GetGitRevisionDescription)
git_describe(VERSION --tags --match [vV][0-9]*)

if(${VERSION} MATCHES ".*NOTFOUND.*")
    set(VERSION "v0.0.0")
endif()

#parse the version information into pieces.
string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_SHA1 "${VERSION}")

option(FORCE_NEW_VERSION "Increase the version number from git by one patach. Enable during development to not mix up versions." OFF)
if(FORCE_NEW_VERSION)
    MATH(EXPR VERSION_PATCH "${VERSION_PATCH}+1")
endif()
set(VERSION_SHORT "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")