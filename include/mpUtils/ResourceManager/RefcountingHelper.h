/*
 * mpUtils
 * RefcountingHelper.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_REFCOUNTINGHELPER_H
#define MPUTILS_REFCOUNTINGHELPER_H

// includes
//--------------------
#include <functional>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief allows the resource wrapper to signal its construction and destruction without
 * knowing about the template parameters of the resource cache
 */
class RefcountingHelper
{
public:
    using HandleType = unsigned int;

    RefcountingHelper(std::function<void(HandleType)> construction, std::function<void(HandleType)> destruction)
            : m_construction(std::move(construction)), m_destruction(std::move(destruction)) {}

    void signalConstruction(HandleType h) { m_construction(h); }
    void signalDestruction(HandleType h) { m_destruction(h); }

private:
    std::function<void(HandleType)> m_construction;
    std::function<void(HandleType)> m_destruction;
};

}

#endif //MPUTILS_REFCOUNTINGHELPER_H
