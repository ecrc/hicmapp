#include <hicmapp/runtime/interface/HicmaContext.hpp>

namespace hicmapp::runtime {

    HicmaContext::HicmaContext() = default;

    const hcorepp::kernels::RunContext &
    HicmaContext::GetMainContext() {
        return hcorepp::kernels::ContextManager::GetInstance().GetContext(0);
    }

    const hcorepp::kernels::RunContext &HicmaContext::GetContext(size_t aIdx) {
        return hcorepp::kernels::ContextManager::GetInstance().GetContext(aIdx);
    }

    size_t HicmaContext::GetNumOfContexts() {
        return hcorepp::kernels::ContextManager::GetInstance().GetNumOfContexts();
    }

    void HicmaContext::SyncMainContext() {
        hcorepp::kernels::ContextManager::GetInstance().SyncMainContext();
    }

    void HicmaContext::SyncContext(size_t aIdx) {
        hcorepp::kernels::ContextManager::GetInstance().SyncContext(aIdx);
    }

    void HicmaContext::SyncAll() {
        hcorepp::kernels::ContextManager::GetInstance().SyncAll();
    }

    HicmaCommunicator &HicmaContext::GetCommunicator() {
        return mCommunicator;
    }

    void HicmaContext::SetCommunicator(HicmaCommunicator &aCommunicator) {
        mCommunicator = aCommunicator;
    }

    HicmaContext::HicmaContext(HicmaCommunicator aComm) : mCommunicator(aComm) {
    }

}