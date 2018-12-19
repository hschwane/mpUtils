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

using namespace mpu;
using namespace std;


//    // type list
//    template <typename ... Types>
//    struct tl {};
//
//
//
//    // prepend
//    template <typename TypeList, typename T>
//    struct tl_prepend{};
//
//    template <typename ... ListItems, typename T>
//    struct tl_prepend<T, tl<ListItems...>>
//    {
//        using type = tl<T, ListItems...>;
//    };
//
//    template <typename TypeList, typename T>
//    using tl_prepend_t = typename tl_prepend<T,TypeList>::type;
//
//
//
//    // append
//    template <typename TypeList, typename T>
//    struct tl_append{};
//
//    template <typename ... ListItems, typename T>
//    struct tl_append<T, tl<ListItems...>>
//    {
//        using type = tl<ListItems...,T>;
//    };
//
//    template <typename TypeList, typename T>
//    using tl_append_t = typename tl_append<T,TypeList>::type;
//
//
//
//    // concatinate
//    template <typename TypeListA, typename TypeListB>
//    struct tl_concat{};
//
//    template <typename ... ListAItems, typename ... ListBItems>
//    struct tl_concat< tl<ListAItems...>, tl<ListBItems...>>
//    {
//        using type = tl<ListAItems...,ListBItems...>;
//    };
//
//    template <typename TypeListA, typename TypeListB>
//    using tl_concat_t = typename tl_concat<TypeListA,TypeListB>::type;
//
//    // head
//    // tail


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









// test function
template <class ...Ts>
void testFunction()
{
    int t[] = {0, ( cout << typeid(Ts).name() << "\n",1)...};
}

template <typename Tpl>
struct foo;

template <class ...Ts>
struct foo< tuple<Ts...>>
{
    void operator()()
    {
        testFunction<Ts...>();
    }
};


template <typename T> class debug_t;

int main()
{
//    using myListA = tl<float,int,double>;
//    using myListB = tl<int,int,int>;
//    using myPrependedList = tl_prepend_t<myListA,long>;
//    using myAppendedList = tl_append_t<myPrependedList,long>;


//    using myListA = tl_make_t<int,double,float,long> ;
//    using myListB = tl_make_t<int,double,float,long> ;
//
//    using tt = std::make_tuple<int,float,double>();
//
    using tl = std::tuple<int,float,double>;

//    debug_t< bla<tl> > dt;
    testFunction<tl>();

//    testFunction<tl::types>();



//    bind_t<tl,testFunction>();

    return 0;
}