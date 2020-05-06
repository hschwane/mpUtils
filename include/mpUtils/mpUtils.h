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
#include "Misc/additionalMath.h"
#if !defined(MPU_NO_PREPROCESSOR_UTILS)
    #include "Misc/preprocessorUtils.h"
    #include "Misc/alignment.h"
#endif
#include "Misc/EmbeddedData.h"
#include "mpUtils/Misc/CallbackHandler.h"
#include "mpUtils/Misc/StateMachine.h"
#include "mpUtils/Misc/CopyMoveAtomic.h"

// image loading
#include "Misc/Image.h"

// configuration util
#include "mpUtils/external/toml/toml.hpp"
#include "mpUtils/Misc/tomlStore.h"

// resource management
#include "mpUtils/ResourceManager/readData.h"
#include "mpUtils/ResourceManager/Resource.h"
#include "mpUtils/ResourceManager/ResourceCache.h"
#include "mpUtils/ResourceManager/ResourceManager.h"
#include "mpUtils/ResourceManager/mpUtilsResources.h"

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

// compiletime math
#include "mpUtils/external/gcem/gcem.hpp"

// tinyfiledialogs
#include "mpUtils/external/tinyfd/tinyfiledialogs.h"

// EnTT entity component system
#include "mpUtils/external/entt/entt.hpp"

// Jakob Progsch thread pool
#include "mpUtils/external/threadPool/ThreadPool.h"

// matrix type might be useful without cuda
#include "Cuda/Matrix.h"
#include "Cuda/MatrixMath.h"

#endif //MPUTILS_MPUTILS_H
