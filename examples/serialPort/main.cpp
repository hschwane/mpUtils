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

#include <mpUtils/mpUtils.h>

using namespace mpu::io;
using namespace std;

int main()
{
//    Log myLog( LogLvl::ALL, ConsoleSink());
//    myLog.printHeader("devTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    SerialPort serial("/dev/ttyUSB0",BaudRate::BAUD_57600);

    serial.write("Hallo\n",6);

    int i = 0;
    char b;
    while(true) {
        std::string s = std::to_string(i) + "\n";
        serial.write(s.c_str(),s.length());
        i++;
//        int n = serial.read(&b,1);
//        if(n==1)
//            std::cout << b;
    }

    return 0;
}