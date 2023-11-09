#include <hicmapp/runtime/interface/RunTimeSingleton.hpp>
#include <iostream>

namespace hicmapp::runtime {

    template<typename T>
    hicmapp::runtime::RunTimeInterface<T> *RunTimeSingleton<T>::runtime_instance = nullptr;

    template<typename T>
    RunTimeSingleton<T>::RunTimeSingleton() = default;

    template<typename T>
    RunTimeInterface<T> *RunTimeSingleton<T>::GetRunTimeInstance() {
        if (runtime_instance != nullptr) {
            return runtime_instance;
        } else {

            throw std::runtime_error("RunTimeSingleton::GetRunTimeInstance, Instance is null.\n");
        }
    }

    template<typename T>
    void RunTimeSingleton<T>::setRunTimeInstance(hicmapp::runtime::HicmaHardware& aHardware) {
        if (runtime_instance == nullptr) {
            runtime_instance = hicmapp::runtime::RunTimeFactory<T>::CreateRunTimeInstance(aHardware);
        }

        if (runtime_instance == nullptr) {
            throw std::bad_alloc();
        }


    }

    HICMAPP_INSTANTIATE_CLASS(RunTimeSingleton);

}