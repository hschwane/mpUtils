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

/*
 * This project is used by the developers to play with and test new features
 */

#include <mpUtils.h>
#include <mpGraphics.h>

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());

    for(int i = 0; i < 10; ++i)
    {
        logINFO("Test") << "Testing log things";
        debugMark();
        sleep(1);
    }

    return 0;
}