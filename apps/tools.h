#pragma once

#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__) || defined(__clang__)
#define __CURRENT_FUNCTION__ __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
#define __CURRENT_FUNCTION__ __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#define __CURRENT_FUNCTION__ __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
#define __CURRENT_FUNCTION__ __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
#define __CURRENT_FUNCTION__ __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#define __CURRENT_FUNCTION__ __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#define __CURRENT_FUNCTION__ __func__
#else
#define __CURRENT_FUNCTION__ "(unknown)"
#endif