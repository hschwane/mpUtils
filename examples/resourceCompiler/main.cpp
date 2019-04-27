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


ADD_RESOURCE(Text, "blubber/hw.txt");

int main()
{
    mpu::Resource text = LOAD_RESOURCE(Text);
    std::cout << std::string(text.data(), text.size()) << std::endl;
//    for(int i =0; i<gTextSize;i++)
//        std::cout << gTextData[i];
//    std::cout << std::endl;
    return 0;
}