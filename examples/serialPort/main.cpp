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

    std::cout << "available ports" << std::endl;
    auto a = listAvailableSerialPorts();
    for(auto&& p : a) {
        std::cout << p << std::endl;
    }

    SerialPortStream sps("/dev/ttyUSB0",BaudRate::BAUD_9600);

    sps << "Hello Serial\n";
    sps << "send an integer: \n";
    int i=-1;
    sps >> i;
    sps << "you sent " << i << "\n";
    sps << "now i will repeat to you line by line. send \"q\" to quit\n";

    std::string s;
    while(true) {
        std::getline(sps,s);
        if(s == "q")
            break;
        else if(!s.empty()) {
            sps << s << "\n";
            std::cout << "received: " << s << "\n";
        }
    }

    return 0;
}