/*
 * mpUtils
 * callbackUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_CALLBACKHANDLER_H
#define MPUTILS_CALLBACKHANDLER_H

// includes
//--------------------
#include <vector>
//--------------------

// includes
//--------------------
namespace mpu {
//--------------------

template<typename F>
class CallbackHandler
{
public:

    using CallbackType = F;
    using IdType = int;
    using CallbackStorage = std::pair<IdType,CallbackType>;
    using CallbacklVector = std::vector<CallbackStorage>;

    IdType addCallback(CallbackType f)
    {
        IdType id = 0;
        if(!m_callbacks.empty())
            id = m_callbacks.back().first+1;
        m_callbacks.emplace_back(id, f);
        return id;
    }

    void removeCallback(IdType id)
    {
        auto it = std::find_if(m_callbacks.begin(), m_callbacks.end(),
                                       [&id](const CallbackStorage& a)
                                       { return id == a.first; });

        if(it != m_callbacks.end())
        {
            if(m_isExecuting)
            {
                it->second = CallbackType();
                m_removedDuringExec = true;
            } else
            {
                m_callbacks.erase(it);
            }
        }
    }

    template <typename  ... TArgs>
    void executeCallbacks(TArgs&& ... args)
    {
        m_isExecuting = true;
        for(int i =0; i<m_callbacks.size(); i++)
        {
            if(m_callbacks[i].second)
                m_callbacks[i].second(std::forward<TArgs>(args)...);
        }
        m_isExecuting = false;

        if(m_removedDuringExec)
        {
            m_callbacks.erase( std::remove_if(m_callbacks.begin(), m_callbacks.end(),
                                    [](const CallbackStorage& a)
                                    { return !(a.second); }),m_callbacks.end());
            m_removedDuringExec = false;
        }
    }

private:
    CallbacklVector m_callbacks;
    bool m_isExecuting{false};
    bool m_removedDuringExec{false};
};





}
#endif //MPUTILS_CALLBACKHANDLER_H
