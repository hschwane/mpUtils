/*
 * mpUtils
 * templateUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_TEMPLATEUTILS_H
#define MPUTILS_TEMPLATEUTILS_H

// includes
//--------------------
#include <type_traits>
#include "type_traitUtils.h"
//--------------------

// this file contains device/host functions that also need to compile when using gcc
//--------------------
#ifndef CUDAHOSTDEV
    #ifdef __CUDACC__
        #define CUDAHOSTDEV __host__ __device__
    #else
        #define CUDAHOSTDEV
    #endif
#endif
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
// base_cast
// Thanks to Francesco Biscani!
namespace detail {
    // A structure that can be implicitly converted to any type.
    struct no_baseclass_flag {};

    // Machinery to cast the input reference x to one of its bases B.
    // This overload is selected only if B is a base of T.
    template<typename B, typename T, std::enable_if_t<std::is_base_of<B, T>::value, int> = 0>
    CUDAHOSTDEV const B &base_cast(const T &x)
    {
        return static_cast<const B &>(x);
    }

    // Overload taken if B is not a base of T. In this case we return
    // an object that can be converted to anything.
    template<typename B, typename T, std::enable_if_t<!std::is_base_of<B, T>::value, int> = 0>
    CUDAHOSTDEV no_baseclass_flag base_cast(const T &)
    {
        return no_baseclass_flag{};
    }
}

/**
 * @brief If B is a base class of x, x will be casted to B and returned as const reference.
 *      Otherwise a no_baseclass_flag will be returned.
 * @tparam B The type of the base class in question.
 * @tparam T The type of x.
 * @param x  A const reference to the object to be casted to type B.
 * @return   A const reference to x casted the type of B or a const reference to a default constructed B
 */
template<typename B, typename T>
CUDAHOSTDEV auto base_cast(const T &x)
{
    return detail::base_cast<B>(x);
};

//-------------------------------------------------------------------
// logical operations (available in std since c++ 17)

/**
 * @brief shorthand for integral constant of type bool
 *      This is available in std since c++17.
 */
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

/**
 * @brief negate B which has a member bool::value
 *       This is available in std since c++17.
 */
template<class B>
struct negation : bool_constant<!bool(B::value)> { };

/**
 * @brief Forms logical conjunction of Bs which have a member bool value
 *      This is available in std since c++17.
 */
template<class...> struct conjunction : std::true_type { };
template<class B1> struct conjunction<B1> : B1 { };
template<class B1, class... Bn>
struct conjunction<B1, Bn...>
        : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template<class... B>
constexpr bool conjunction_v = conjunction<B...>::value;

/**
 * @brief Forms logical disjunction of Bs which have a member bool value
 *      This is available in std since c++17.
 */
template<class...> struct disjunction : std::false_type { };
template<class B1> struct disjunction<B1> : B1 { };
template<class B1, class... Bn>
struct disjunction<B1, Bn...>
        : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>>  { };

template<class... B>
constexpr bool disjunction_v = disjunction<B...>::value;

/**
 * @brief Has member type = T if b is true, has member type = F if not.
 * @tparam b boolean constexpr to switch on
 * @tparam T result if b == true
 * @tparam F result if F == false
 */
template <bool b, typename T, typename F>
struct if_else
{
    using type = T;
};

template <typename T, typename F>
struct if_else<false, T, F>
{
    using type = F;
};

//!< Has member type = T if b is true, has member type = F if not. Shorthand for if_else::type.
template <bool b, typename T, typename F>
using if_else_t = typename if_else<b,T,F>::type;

//!< this is always false but depends on template arguments (so it is false when the template is instantiated)
template <typename...>
struct always_false : std::false_type {};

//!< this is always false but depends on template arguments (so it is false when the template is instantiated)
template <typename... Ts>
static constexpr bool always_false_v = always_false<Ts...>::value;

//-------------------------------------------------------------------
// operations / helper on std::tuple

/**
 * @brief the type of a tuple that results from the concatenation of tuples of the types in Ts
 */
template<typename...Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

/**
 * @brief tuple with all the types in Ts for which cond<T>::value is true
 */
template<template<class T> class cond, typename...Ts>
using remove_t = tuple_cat_t<
        typename std::conditional<
                cond<Ts>::value,
                std::tuple<Ts>,
                std::tuple<>
        >::type...
>;

/**
 * @brief find the index of the first occurence of T in Tuple
 * @tparam T a type that is part of Tuple
 * @tparam Tuple The tuple to look for
 */
template <class T, class Tuple>
struct index_of;

template <class T, class... Types>
struct index_of<T, std::tuple<T, Types...>>
{
    static constexpr std::size_t value = 0;
};

template <class T, class U, class... Types>
struct index_of<T, std::tuple<U, Types...>>
{
    static_assert(sizeof...(Types)>0, "Type T was not found in Tuple!");
    static constexpr std::size_t value = 1 + index_of<T, std::tuple<Types...>>::value;
};

//!< Find the index of the first occurence of T in Tuple. Shorthand for index_of::type.
template <class T, class Tuple>
static constexpr size_t index_of_v = index_of<T,Tuple>::value;

/**
 * @brief instantiate template Class with types from tuple
 * @tparam Class the class template to instantiate
 * @tparam Tuple the std::tuple from where to get the types
 */
template <template<typename ...>class Class, typename Tuple>
struct instantiate_from_tuple;

template <template<typename ...>class Class, typename ...TupleTs>
struct instantiate_from_tuple<Class, std::tuple<TupleTs...>>
{
    using type = Class<TupleTs...>;
};

//!< instantiate template Class with types from std::tuple Tuple
template <template<typename ...>class Class, typename Tuple>
using instantiate_from_tuple_t = typename instantiate_from_tuple<Class,Tuple>::type;

/**
 * @brief Has Tuple type T? Will evaluate to std::true_type if T is in Tuple, std::false_type otherwise
 */
template <typename T, typename Tuple>
struct has_type;

template <typename T>
struct has_type<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};

//!< shorthand for has_type<>::type
template <typename T, typename Tuple>
using has_type_t = typename has_type<T, Tuple>::type;

//!< shorthand for has_type<>::value
template <typename T, typename Tuple>
static constexpr bool has_type_v = has_type<T, Tuple>::value;

//-------------------------------------------------------------------
// operations / helper on std::integer_list

// check if integer list is sorted from lowest to highest and has no duplicate entries
/**
 * @brief Check if integer list List is sorted from lowest to highest and has no duplicate entries.
 *      inherits from std::true_type or std::false_type depending on result
 * @tparam List the std::integer::list to check
 */
template <typename List, typename Enabler = void>
struct is_strict_rising : std::false_type {};

template <typename T, T only>
struct is_strict_rising< std::integer_sequence<T,only>> : std::true_type {};

template <typename T, T prev, T first, T ...Ints>
struct is_strict_rising< std::integer_sequence<T, prev, first, Ints...>, typename std::enable_if<prev<first>::type > : is_strict_rising<std::integer_sequence< T, first, Ints...>> {};

//!< Check if integer list List is sorted from lowest to highest and has no duplicate entries. Shorthand for is_strict_rising::type.
template <typename List>
static constexpr bool is_strict_rising_v = is_strict_rising<List>::value;

/**
 * @brief concatenate multiple std::integer_sequence
 * @tparam T a number of sequences to concatenate
 */
template <typename ...T>
struct is_concat;

template <typename T, T... p1, T... p2, typename ...Tail>
struct is_concat<std::integer_sequence<T, p1...>, std::integer_sequence<T, p2...>, Tail... >
{
    using first_two = std::integer_sequence<T, p1..., p2...> ;
    using type = typename is_concat<first_two, Tail...>::type;
};

template <typename T, T... elems>
struct is_concat<std::integer_sequence<T, elems...> >
{
    using type = std::integer_sequence<T, elems...>;
};

//!< Concatenate multiple std::integer_sequence. Shorthand for is_concat::type.
template <typename ...T>
using is_concat_t = typename  is_concat<T...>::type;


/**
 * @brief Filters a integer sequence using the predicate Pred. The result is a new std::integer_sequence containing only
 *          those entries for which Pred<Type,entry>::value is true.
 * @tparam Pred a template with a bool member "value" that is true if the entry of type U should be included in the new list.
 * @tparam Data the std::integer_sequence to filter
 */
template <template <typename U, U> class Pred, typename Data>
struct is_filter;

template <template <typename U, U> class Pred, typename T, T head, T... tail>
struct is_filter<Pred, std::integer_sequence<T, head, tail...> >
{
    using tlist = std::integer_sequence<T, tail...>;
    using ftail = typename is_filter<Pred, tlist>::type;
    using h = std::integer_sequence<T, head>;
    using type = typename if_else<Pred<T, head>::value, typename is_concat<h, ftail>::type, ftail>::type;
};

template <template <typename U, U> class Pred, typename T>
struct is_filter<Pred, std::integer_sequence<T> >
{
    using type = std::integer_sequence<T>;
};

//!< Filter Data using predicate Pred. Shorthand for is_filter::type
template <template <typename U, U> class Pred, typename Data>
using is_filter_t = typename is_filter<Pred,Data>::type;

/**
 * @brief Bind a compare-template to a predicate to use with is_filter
 * @tparam C the compare template to bind to the predicate
 * @tparam T type of the values that will be compared
 * @tparam t the value to which to compare
 */
template <template <typename U, U, U> class C,typename T, T t>
struct is_filter_bind_pred
{
    template <typename V, V v>
    struct pred
    {
        static const bool value = C<T, v, t>::value;
    };
};

/**
 * @brief Negate a predicate to use with is_filter
 * @tparam Pred
 */
template <template <typename U, U> class Pred>
struct is_filter_neg_pred
{
    template <typename T, T v>
    struct pred
    {
        static const bool value = !Pred<T, v>::value;
    };
};

/**
 * @brief comparison-template, true if l < r
 * @tparam T type of operands
 * @tparam l first operand
 * @tparam r second operand
 */
template <typename T, T l, T r>
struct l_smaller_r
{
    static const bool value = l < r;
};

/**
 * @brief comparison-template, true if l > r
 * @tparam T type of operands
 * @tparam l first operand
 * @tparam r second operand
 */
template <typename T, T l, T r>
struct l_bigger_r
{
    static const bool value = l > r;
};

/**
 * @brief Performs Quicksort on std::integer_sequence
 * @tparam Comp comparison-template to be used during sorting
 * @tparam Data the std::integer_sequence to sort
 */
template <template <typename T, T, T> class Comp,typename Data>
struct is_q_sort;

template <template <typename U, U, U> class Comp,typename T, T head, T... tail>
struct is_q_sort<Comp, std::integer_sequence<T, head, tail...> >
{
    // make a predicate that will use the compare function comp to compare v to head
    template <typename V, V v>
    struct pred : is_filter_bind_pred<Comp, V, v>::template pred<T, head> {};

    // like pred, but inverted
    template <typename V, V v>
    struct npred : is_filter_neg_pred<pred>::template pred<V, v> {};


    using fl = typename is_filter<npred, std::integer_sequence<T, tail...> >::type; // find everything that compares negative to head
    using tl = typename is_filter< pred, std::integer_sequence<T, tail...> >::type; // everything that compares positive with head
    using sfl = typename is_q_sort<Comp, fl>::type; // list of everything that compares negative to head, but recursively sorted
    using stl = typename is_q_sort<Comp, tl>::type; // list of everything that compares positive to head, but recursively sorted
    using hl =  std::integer_sequence<T, head>; // head as a list
    using type = typename is_concat<sfl, hl, stl>::type; // reassemble the list
};

template <template <typename U, U, U> class Pred,typename T>
struct is_q_sort<Pred, std::integer_sequence<T> >
{
    using type = std::integer_sequence<T>;
};

//!< sort std::integer_sequence Data in ascending order
template <typename Data>
using is_sort_asc_t = typename is_q_sort<l_smaller_r,Data>::type;

//!< sort std::integer_sequence Data in descending order
template <typename Data>
using is_sort_desc_t = typename is_q_sort<l_bigger_r,Data>::type;

/**
 * @brief removes duplicates from a sorted std::integer_sequence !! must be sorted first !!
 * @tparam List the std::integer_sequence to remove duplicates from
 */
template <typename List>
struct is_rm_duplicates;

template <typename T, T only>
struct is_rm_duplicates< std::integer_sequence<T,only>>
{
    using type = std::integer_sequence<T,only>;
};

template <typename T>
struct is_rm_duplicates< std::integer_sequence<T>>
{
    using type = std::integer_sequence<T>;
};

template <typename T, T first, T second, T ...Ints>
struct is_rm_duplicates< std::integer_sequence<T,first,second,Ints...>>
{
    using tail = std::integer_sequence<T,second,Ints...>;
    using recursion_t = typename is_rm_duplicates<tail>::type;
    using head_t = if_else_t< first != second, std::integer_sequence<T,first>, std::integer_sequence<T> >;
    using type = is_concat_t<head_t, recursion_t>;
};

//!< Removes duplicates from a sorted std::integer_sequence !! must be sorted first !! Shorthand for is_rm_duplicates::type
template <typename List>
using is_rm_duplicates_t = typename is_rm_duplicates<List>::type;


//-------------------------------------------------------------------
// misc

/**
 * @brief has static member ::value which is the number of arguments of function a function type T
 */
template <typename T>
struct argumentCount : argumentCount<decltype(&T::operator())> {};

template <typename R, typename... Args>
struct argumentCount<R(*)(Args...)> : std::integral_constant<unsigned, sizeof...(Args)> {};

template <typename R, typename C, typename... Args>
struct argumentCount<R(C::*)(Args...)> : std::integral_constant<unsigned, sizeof...(Args)> {};

template <typename R, typename C, typename... Args>
struct argumentCount<R(C::*)(Args...) const> : std::integral_constant<unsigned, sizeof...(Args)> {};

/**
 * @brief the number of arguments of function a function type
 */
template <typename T>
constexpr size_t argumentCount_v = argumentCount<T>::value;

/**
 * @brief returns the number of arguments of function a function
 */
template <typename func>
constexpr size_t getArgumentCount(func)
{
    return argumentCount_v<func>;
}


}

#endif //MPUTILS_TEMPLATEUTILS_H
