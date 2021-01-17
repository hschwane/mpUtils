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
#include <mpUtils/mpGraphics.h>

using namespace mpu;
using namespace std;

int main()
{
//    Log myLog( LogLvl::ALL, ConsoleSink());
//    myLog.printHeader("devTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    toml::value v1(true);     // boolean
    toml::value v2(42);       // integer
    toml::value v3(3.14);     // floating
    toml::value v4("foobar"); // string
    toml::value v5(toml::local_date(2019, toml::month_t::Apr, 1)); // date
    toml::value v6{1, 2, 3, 4, 5};                                 // array

    toml::value table{ {"boolean",v1}, {"integer",v2}, {"floating",v3}, {"string",v4}, {"date", v5}, {"array",v6}};
    toml::value table2{ {"test", table} };

    std::cout << std::setw(20) << table2;

    toml::value dynamic;
    dynamic["Test"] = 42;
    dynamic["string"] = "Hallo";
    dynamic["hallo"] = true;
    dynamic["42"] = "test";

    dynamic["blubber"]["int"] = 42;
    dynamic["blubber"]["subBlubber"]["int"] = 21;
    dynamic["blubber"]["subBlubber"]["subsubBlubber"]["int"] = 10.5f;
    std::cout << std::setw(20) << dynamic;

    glm::vec4 glv(0,1,2,3);
    toml::value vec({glv[0],glv[1],glv[2],glv[3]});
    std::cout << std::setw(20) << vec << std::endl;


    toml::store("/home/hendrik/test/test.toml", table2);
    toml::value fromFile = toml::parse("/home/hendrik/test/test.toml");

    std::cout << toml::find(fromFile,"test","string") << std::endl;
    std::cout << toml::find_or<std::string>(fromFile["test"],"stringire","missing");
    std::cout << toml::find_or<std::string>(fromFile["testere"],"string","missing");
    std::cout << std::endl;

    std::vector<std::string> vos = {"blub", "bla"};
    toml::value tvos(vos);
    toml::value table3{{"vos",tvos}};

    auto v = toml::find<std::vector<std::string>>(table3,"vos");

    std::cout << v[1];

    std::cout << std::setw(20) << tvos;

    return 0;
}