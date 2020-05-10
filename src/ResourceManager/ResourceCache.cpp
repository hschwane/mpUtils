/*
 * mpUtils
 * ResourceCache.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#include "mpUtils/ResourceManager/ResourceManager.h"
thread_local bool mpu::ReloadMode::enabled(false);