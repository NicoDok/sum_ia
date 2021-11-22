#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <stdexcept>

#include <mathlib/matrix.h>
#include <mathlib/vector.h>

TEST_CASE("Check constr", "[vector]") {
    REQUIRE_NOTHROW(MathLib::vector<int>(static_cast<uint64_t>(5)));
    REQUIRE_NOTHROW(MathLib::vector<int>(5));
    REQUIRE_NOTHROW(MathLib::vector<float>(5));
    REQUIRE_NOTHROW(MathLib::vector<char>(5));
    //REQUIRE_NOTHROW(MathLib::vector<char>{3, 2, 1});
}

TEST_CASE("Check assign", "[vector]") {
    MathLib::vector<int> tab_int(static_cast<uint64_t>(5));
    tab_int[3] = 2;

    REQUIRE (tab_int[3] == 2);
}

TEST_CASE("Check exception oversize", "[vector]") {
    REQUIRE_THROWS_AS(MathLib::vector<int>(std::numeric_limits<uint64_t>::max()), std::bad_array_new_length);
}

TEST_CASE("vectors can be sized and resized", "[vector]") {
    MathLib::vector<int> v(5);

    REQUIRE( v.size() == 5 );
    REQUIRE( v.capacity() >= 5 );

    SECTION( "resizing bigger changes size and capacity" ) {
        v.resize( 10 );

        REQUIRE( v.size() == 10 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "resizing smaller changes size but not capacity" ) {
        v.resize( 0 );

        REQUIRE( v.size() == 0 );
        REQUIRE( v.capacity() >= 5 );
    }
    SECTION( "reserving bigger changes capacity but not size" ) {
        v.reserve( 10 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "reserving smaller does not change size or capacity" ) {
        v.reserve( 0 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );
    }
    SECTION( "reserving bigger changes capacity but not size" ) {
        v.reserve( 10 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 10 );
        SECTION( "reserving down unused capacity does not change capacity" ) {
            v.reserve( 7 );
            REQUIRE( v.size() == 5 );
            REQUIRE( v.capacity() >= 10 );
        }
    }
}
/*
TEST_CASE("Vector of vector", "[vector]") {
    MathLib::vector<MathLib::vector<int>> matrice{ { 1, 2, 3 }, 
                        { 4, 5, 6 }, 
                        { 7, 8, 9, 4 } }; 
    matrice[2][3];
}
*/