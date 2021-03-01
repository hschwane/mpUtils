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

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main(int argc, const char* argv[])
{
    struct Opts {
        bool b{};
        bool b2{};
        int i{};
        float f{};
        double d{};
        std::string s{};
    };

    auto options = cfg::ArgParser<Opts>::parse(argc,argv,{
            {"bool","b", &Opts::b, "A bool for testing.", ""},
            {"bool2","c", &Opts::b2, "A second bool for testing.", ""},
            {"integer","i", &Opts::i, "An integer for testing.", "<integer>"},
            {"float","f", &Opts::f, "A float for testing.", "<number>"},
            {"double","d", &Opts::d, "A double for testing.", "<number>"},
            {"string","s", &Opts::s, "A string for testing.", "<string>"},
        }, "devText [options] <positional arguments> \nTesting the mpUtils argument parser.",
        "mpUtils library test v" MPU_VERSION_STRING "\nCopyright: Hendrik Schwanekamp\nProvided under MIT license");

    std::cout << "\n\nFound options:\n"
              << std::boolalpha << "bool: " << options.b
              << "\nbool2: " << options.b2
              << "\nint: " << options.i
              << "\nfloat: " << options.f
              << "\ndouble: " << options.d
              << "\nstring: " << options.s
              << "\n";

    std::cout << "\npositionals: \n";
    for(auto&& s : options.positional) {
        std::cout << s << "\n";
    }

    return 0;
}