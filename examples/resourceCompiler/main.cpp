/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

#include <iostream>
#include "mpUtils/mpUtils.h"

int main()
{
    mpu::Resource text = LOAD_RESOURCE(blubber_hw_txt);
    std::cout << std::string(text.data(), text.size()) << std::endl;
    return 0;
}