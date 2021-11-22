#pragma once

#include <cstdint>

namespace MathLib {

//uint64_t next_pow2(uint64_t x) {
//	return x == 1 ? 1 : 1<<(64-__builtin_clzl(x-1));
//}

uint64_t next_pow2(uint64_t x) {
	x |= x>>1;
	x |= x>>2;
	x |= x>>4;
	x |= x>>8;
	x |= x>>16;
	x |= x>>32;
	return x;
}

}