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
 * This is testing features of the matrix class... to be replaced by actual unit tests in the future...
 */

#include <mpUtils/mpUtils.h>
//#include <mpUtils/mpGraphics.h>
//#include <glm/glm.hpp>
//#include <mpUtils/mpCuda.h>

//    // type list
//    struct tl_end_t {};
//
//    template <typename T, typename U>
//    struct tl
//    {
//        using head=T;
//        using tail=U;
//    };
//
//
//
//    // creation of type list
//    template <typename ... Ts> struct tl_make;
//
//    // case 1: consume one type per call
//    template <typename T, typename ... REST>
//    struct tl_make<T, REST...>
//    {
//        using type = tl<T, typename tl_make<REST...>::type>;
//    };
//
//    // case 2: abort recursion when all types are consumed
//    template <>
//    struct tl_make<> { using type=tl_end_t; };
//
//    template <typename ... Ts>
//    using tl_make_t = typename tl_make<Ts...>::type;
//
//
//
//    // prepend
//    template <typename T, typename RecursiveList>
//    using tl_prepend_t = tl<T, RecursiveList>;
//
//
//
//    // append
//    template <typename T, typename TList>
//    struct tl_append;
//
//    // (tl_end_t item empty list) -> Still an empty list
//    template <>
//    struct tl_append<tl_end_t, tl_end_t> { using type = tl_end_t; };
//
//    // (T, empty list) -> List which only contains T
//    // This is usually the recursion abort step when adding an item to a list
//    template <typename T>
//    struct tl_append<T, tl_end_t> { using type = tl<T, tl_end_t>; };
//
//    // (T, list) -> Recurse until tail of list, and return a version with T at its end
//    template <typename Head, typename Rest, typename T>
//    struct tl_append< T, tl<Head, Rest>> { using type = tl<Head, typename tl_append<T, Rest>::type>; };
//
//    // (tl_end_t, list) -> Do nothing
//    template <typename Head, typename Rest>
//    struct tl_append< tl_end_t, tl<Head, Rest>> { using type = tl<Head, Rest>; };
//
//    template <typename TList, typename T>
//    using tl_append_t = typename tl_append<TList, T>::type;
//
//
//
//    // concat ListA to ListB
//    template <typename TListA, typename TListB>
//    struct tl_concat;
//
//    // (empty list and empty list) -> Still an empty list
//    template <>
//    struct tl_concat<tl_end_t, tl_end_t> { using type = tl_end_t; };
//
//    // (ListB, empty list) -> ListB
//    // This is usually the recursion abort step when adding a list to a list
//    template <typename Head, typename Rest>
//    struct tl_concat< tl<Head, Rest>, tl_end_t> { using type = tl<Head, Rest>; };
//
//    // (ListA, ListB) -> Recurse until tail of list, and return a version with T at its end
//    template <typename Head, typename Rest, typename ListA>
//    struct tl_concat< ListA, tl<Head, Rest>>
//    {
//        using type = tl<Head, typename tl_concat<ListA, Rest>::type>;
//    };
//
//    // (empty list, ListB) -> ListB
//    template <typename Head, typename Rest>
//    struct tl_concat< tl_end_t, tl<Head, Rest>>
//    {
//        using type = tl<Head, Rest>;
//    };
//
//    template <typename TList, typename T>
//    using tl_concat_t = typename tl_concat<TList, T>::type;
//
//
//
//    // head
//    template <typename TList>
//    struct tl_head;
//
//    template <typename Head, typename Tail>
//    struct tl_head< tl<Head,Tail> >
//    {
//        using type = Head;
//    };
//
//    template <typename TList>
//    using tl_head_t = typename tl_head<TList>::type;
//
//
//
//    // tail
//    template <typename TList> struct tl_tail;
//
//    template <typename Head, typename Rest>
//    struct tl_tail< tl<Head,Rest> >
//    {
//        using type = typename tl_tail<Rest>::type;
//    };
//
//    template <typename T>
//    struct tl_tail< tl<T,tl_end_t> >
//    {
//        using type = T;
//    };
//
//    template <typename TList>
//    using tl_tail_t = typename tl_tail<TList>::type;


//    struct empty_tuple;
//
//    template <typename Tuple>
//    struct pop_head
//    {
//        using head = void;
//        using rest = std::tuple<void>;
//    };
//
//    template <typename Head, typename ...Ts>
//    struct pop_head <std::tuple<Head, Ts...>>
//    {
//        using head = Head;
//        using rest = std::tuple<Ts...>;
//    };
//
//    template <typename Tuple>
//    using front_t = typename pop_head<Tuple>::head;
//
//    template <typename Tuple>
//    using pop_front_t = typename pop_head<Tuple>::rest;

//using namespace mpu;
//using namespace std;

// -------------------------------------------
// build a index list for the types in Tuple based of the position of the same type in Reference
template <typename Tuple, typename Reference>
struct build_comp_index_list;

template <typename ... Ts, typename Reference>
struct build_comp_index_list <std::tuple<Ts...>, Reference>
{
using type = std::index_sequence< mpu::index_of_v<Ts, Reference>... >;
};

template <typename Tuple, typename Reference>
using build_comp_index_list_t = typename build_comp_index_list<Tuple,Reference>::type;

// -------------------------------------------
// check if the order of types in Tuple is the same as in Reference and there are no duplicates
template <typename Tuple, typename Reference>
using checkOrder = mpu::is_strict_rising< build_comp_index_list_t<Tuple,Reference> >;

// -------------------------------------------
// build tuple using indices and reference tuple, for each number from the list is used as an index, then the type at
// that index in tuple Reference is used to build the new tuple
template <typename Reference, typename IndexList>
struct make_tpl;

template <typename Reference, typename T, T first, T ... Ints>
struct make_tpl<Reference, std::integer_sequence<T,first,Ints...>>
{
using head_tuple = typename std::tuple< typename std::tuple_element<first,Reference>::type>; // do work
using tail_tuple = typename make_tpl<Reference,std::integer_sequence<T,Ints...>>::type; // recursive call
using type = mpu::tuple_cat_t< head_tuple , tail_tuple >; // put together
};

template <typename Reference, typename T>
struct make_tpl<Reference, std::integer_sequence<T>>
{
using type = std::tuple<>;
};

template <typename IndexList, typename Reference>
using make_tpl_t = typename make_tpl<Reference,IndexList>::type;


// -------------------------------------------
// reorder tuple following the reference and remove duplicates
template <typename Tuple, typename Reference>
using reorderd_t = make_tpl_t< mpu::is_rm_duplicates_t< mpu::is_sort_asc_t< build_comp_index_list_t<Tuple,Reference>>>, Reference>;


using order = std::tuple<int,float,double,long>;
using test = std::tuple<float,double,int,int>;

//static_assert(checkOrder< std::tuple<int,float,double>, order >::value, "Wrong type order!"); // should be ok
//static_assert(checkOrder< std::tuple<int,double>, order >::value, "Wrong type order!"); // should be ok
//static_assert(checkOrder< std::tuple<float,double>, order >::value, "Wrong type order!"); // should be ok
//static_assert(checkOrder< std::tuple<float>, order >::value, "Wrong type order!"); // should be ok
//
//static_assert(checkOrder< std::tuple<float,int>, order >::value, "Wrong type order!"); // should fail
//static_assert(checkOrder< std::tuple<double,float>, order >::value, "Wrong type order!"); // should fail
//static_assert(checkOrder< std::tuple<double,double>, order >::value, "Wrong type order!"); // should fail

//template <typename T> class debug_t;
//debug_t< reorderd_t<test,order> > dt;
//debug_t< build_comp_index_list_t< std::tuple<double,float,int,float, double>, order> > dt;

template <typename T>
struct get_arity : get_arity<decltype(&T::operator())> {};

template <typename R, typename... Args>
struct get_arity<R(*)(Args...)> : std::integral_constant<unsigned, sizeof...(Args)> {};
// Possibly add specialization for variadic functions
// Member functions:
template <typename R, typename C, typename... Args>
struct get_arity<R(C::*)(Args...)> :
        std::integral_constant<unsigned, sizeof...(Args)> {};
template <typename R, typename C, typename... Args>
struct get_arity<R(C::*)(Args...) const> :
        std::integral_constant<unsigned, sizeof...(Args)> {};

struct bar
{
    void foo(int a, int b)
    {

    }
};

void blub(int i)
{

}

//template <typename func>
//constexpr size_t noa(func f)
//{
//    return get_arity<func>::value;
//}

//template <typename R, typename C, typename ... Types>
//constexpr std::integral_constant<unsigned, sizeof ...(Types)> getArgumentCount( R(C::*)(Types ...) const)
//{
//    return std::integral_constant<unsigned, sizeof ...(Types)>{};
//}



int main()
{

    return 0;
}