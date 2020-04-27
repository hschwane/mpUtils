/*
 * mpUtils
 * StateMachine.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the StateMachine class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_STATEMACHINE_H
#define MPUTILS_STATEMACHINE_H

// includes
//--------------------
#include <memory>
#include <unordered_map>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------


//-------------------------------------------------------------------
/**
 * class StateMachine
 *
 * Stores and manages states.
 * StateBaseT needs to support onDeactivation() and onActivation() functions,
 * as well as a setStateMachine(this type)  function, which is called between
 * the constructor and onActivate.
 *
 */
template <typename IdT, typename StateBaseT>
class StateMachine
{
public:
    using StateBaseType = StateBaseT;
    using IdType = IdT;

    //!< create state and add it to the state machine
    template <typename StateType, typename ... Ts>
    StateBaseType* createState(IdType id, Ts...args)
    {
        auto result = m_states.emplace(id,std::make_unique<StateType>(std::forward<Ts>(args)...));
        result.first->second->setStateMachine(this);
        return result.first->second.get();
    }

    //!< remove state from the state machine
    void removeState(IdType id)
    {
        auto elem = m_states.find(id);
        if(elem != m_states.end())
        {
            if(m_currentState == elem->second)
                m_currentState = nullptr;
            m_states.erase(elem);
        }
    }

    StateBaseType* getCurrentState() const {return m_currentState;} //!< returns the current state
    StateBaseType* getState(IdType id) const {return m_states[id].get();} //!< returns a specific state if it exists (Be careful!)

    //!< switch state to state with "id"
    void switchState(IdType id)
    {
        if(m_currentState)
            m_currentState->onDeactivation();
        m_currentState = m_states[id].get();
        m_currentState->onActivation();
    }

private:
    StateBaseType* m_currentState{nullptr};
    std::unordered_map<IdType,std::unique_ptr<StateBaseType>> m_states;
};

}
#endif //MPUTILS_STATEMACHINE_H
