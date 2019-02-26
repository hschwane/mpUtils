/*
 * mpUtils
 * mpUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPUTILS_H
#define MPUTILS_MPUTILS_H

// version and path
#include "mpUtils/version.h"
#if !defined(MPU_NO_PATHS)
    #include "mpUtils/paths.h"
#endif

// general stuff
#include "Misc/stringUtils.h"
#include "Misc/timeUtils.h"
#include "Misc/type_traitUtils.h"
#include "Misc/templateUtils.h"
#include "Misc/Range.h"
#include "Misc/pointPicking.h"
#if !defined(MPU_NO_PREPROCESSOR_UTILS)
    #include "Misc/preprocessorUtils.h"
#endif

// configuration util
#include "Cfg/CfgFile.h"

// the logger
#include "Log/ConsoleSink.h"
#include "Log/FileSink.h"
#include "Log/Log.h"
#ifdef __linux__
    #include "Log/SyslogSink.h"
#endif

// timer
#include "Timer/AsyncTimer.h"
#include "Timer/DeltaTimer.h"
#include "Timer/Stopwatch.h"
#include "Timer/Timer.h"

// image processing
#include "mpUtils/external/stb_image.h"

// compiletime math
#include "mpUtils/external/gcem/gcem.hpp"

// matrix type might be useful without cuda
#include "Cuda/Matrix.h"
#include "Cuda/MatrixMath.h"

#endif //MPUTILS_MPUTILS_H
