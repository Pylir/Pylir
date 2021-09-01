#include <catch2/catch.hpp>

#include <pylir/Support/BigInt.hpp>

TEST_CASE("BigInt init", "[BigInt]")
{
    SECTION("double")
    {
        pylir::BigInt integer(3.5);
        CHECK(integer.toString() == "3");
        integer = pylir::BigInt(-3.5);
        CHECK(integer.toString() == "-3");
    }
    SECTION("negative integer")
    {
        pylir::BigInt integer(-3);
        CHECK(integer.toString() == "-3");
    }
    SECTION("From string")
    {
        pylir::BigInt integer("23245354678756453234567898765");
        CHECK(integer.toString() == "23245354678756453234567898765");
    }
}

TEST_CASE("BigInt unary ops", "[BigInt]")
{
    pylir::BigInt integer("23245354678756453234567898765");
    SECTION("Pre increment")
    {
        CHECK((++integer).toString() == "23245354678756453234567898766");
        CHECK(integer.toString() == "23245354678756453234567898766");
    }
    SECTION("Post increment")
    {
        CHECK((integer++).toString() == "23245354678756453234567898765");
        CHECK(integer.toString() == "23245354678756453234567898766");
    }
    SECTION("Pre decrement")
    {
        CHECK((--integer).toString() == "23245354678756453234567898764");
        CHECK(integer.toString() == "23245354678756453234567898764");
    }
    SECTION("Post decrement")
    {
        CHECK((integer--).toString() == "23245354678756453234567898765");
        CHECK(integer.toString() == "23245354678756453234567898764");
    }
    SECTION("Negate")
    {
        CHECK((-integer).toString() == "-23245354678756453234567898765");
        CHECK((-integer).isNegative());
        CHECK_FALSE((-(-integer)).isNegative());
    }
    SECTION("Complement")
    {
        CHECK((~integer).toString() == "-23245354678756453234567898766");
    }
}

TEST_CASE("BigInt bin ops", "[BigInt]")
{
    pylir::BigInt a("435467654");
    pylir::BigInt b("234567");
    CHECK((a + b).toString() == "435702221");
    CHECK((a - b).toString() == "435233087");
    CHECK((a * b).toString() == "102146341195818");
    CHECK((a / b).toString() == "1856");
    CHECK((a % b).toString() == "111302");
    auto [div, mod] = a.divmod(b);
    CHECK(div.toString() == "1856");
    CHECK(mod.toString() == "111302");
    CHECK(pylir::pow(b, 3).toString() == "12906269823562263");
}

TEST_CASE("BigInt comparison", "[BigInt]")
{
    auto two = pylir::BigInt(2);
    auto one = pylir::BigInt(1);

    SECTION("Less")
    {
        CHECK_FALSE(two < one);
        CHECK_FALSE(two < two);
        CHECK(one < two);
    }
    SECTION("Less Equal")
    {
        CHECK_FALSE(two <= one);
        CHECK(two <= two);
        CHECK(one <= two);
    }
    SECTION("Greater")
    {
        CHECK(two > one);
        CHECK_FALSE(two > two);
        CHECK_FALSE(one > two);
    }
    SECTION("Greater Equal")
    {
        CHECK(two >= one);
        CHECK(two >= two);
        CHECK_FALSE(one >= two);
    }
    SECTION("Equal")
    {
        CHECK_FALSE(two == one);
        CHECK(two == two);
        CHECK_FALSE(one == two);
    }
    SECTION("Not Equal")
    {
        CHECK(two != one);
        CHECK_FALSE(two != two);
        CHECK(one != two);
    }
}

TEST_CASE("BigInt getter", "[BigInt]")
{
    SECTION("Signed")
    {
        auto value =
            GENERATE(std::numeric_limits<std::ptrdiff_t>::lowest(), std::numeric_limits<std::ptrdiff_t>::max());
        auto number = pylir::BigInt(value);
        auto result = number.tryGetInteger<std::ptrdiff_t>();
        REQUIRE(result);
        CHECK(*result == value);
    }
    SECTION("Unsigned")
    {
        auto value = std::numeric_limits<std::size_t>::max();
        auto number = pylir::BigInt(value);
        auto result = number.tryGetInteger<std::size_t>();
        REQUIRE(result);
        CHECK(*result == value);
    }
    CHECK_FALSE((++pylir::BigInt(std::numeric_limits<std::size_t>::max())).tryGetInteger<std::size_t>());
    CHECK_FALSE((++pylir::BigInt(std::numeric_limits<std::ptrdiff_t>::max())).tryGetInteger<std::ptrdiff_t>());
    CHECK_FALSE((--pylir::BigInt(std::numeric_limits<std::ptrdiff_t>::lowest())).tryGetInteger<std::ptrdiff_t>());
}
