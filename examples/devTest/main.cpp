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

class BaseState
{
public:
    void onActivation()
    {
        logINFO("State") << "activated";
    }

    void onDeactivation()
    {
        logINFO("State") << "deactivated";
    }

    virtual void blub()=0;

    void setStateMachine(mpu::StateMachine<int,BaseState>* sm)
    {

    }
};

class S1 : public BaseState
{
public:
    void blub() override
    {
        logINFO("") << 1;
    }
};

class S2 : public BaseState
{
public:
    void blub() override
    {
        logINFO("") << 2;
    }
};

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("devTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    mpu::StateMachine<int,BaseState> sm;

    sm.createState<S1>(0);
    sm.createState<S2>(1);

    sm.switchState(0);
    sm.getCurrentState()->blub();
    sm.switchState(1);
    sm.getCurrentState()->blub();

    return 0;
}