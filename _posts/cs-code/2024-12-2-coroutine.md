---
layout:       post
title:        "协程详解"
author:       "Sutdown"
header-style: text
catalog:      true
tags:
    - coroutine
    - C++
---

# 协程详解



## 前言

最近在学协程库，因为计划是12月底之前，时间有限想着直接从代码起步，果然这样子还是不好的，这几天很多东西处于一种似懂非懂的状态，很难受。所以决定写篇文章重新捋一下，之前也写过一点点，但是太浅了。

主要参考[有没有C++大佬把C++20的协程讲解下？ - 知乎](https://www.zhihu.com/question/625089836/answer/3243736805)这个问题中[南山烟雨珠江潮 - 知乎](https://www.zhihu.com/people/frmf)的回答，以及代码随想录的[协程库源码](https://github.com/youngyangyang04/coroutine-lib/tree/main)。



## 正文



### 进程线程协程

> 进程，线程，协程之间的关系，出现的原因，历史渊源等

理解进程和线程从基本概念着手，

**进程**可以简单认为是由程序代码，相关数据还有进程控制块组成的。操作系统基本职责是控制进程的执行，这包括交替执行的方式以及为进程分配资源。

倘若我们要执行一个任务，如果全由进程从头至尾执行那必然效率一般，会思考如果将任务分成不同部分，交由不同的进程并行执行能不能提高效率？进程间的地址空间一般来说都是独立不能互相访问的，如果想要通信必须经过内核，那很明显执行一个任务多次进入内核是得不偿失的。因此出现了**线程**，线程是由线程id，程序计数器，寄存器集合和栈组成，一个进程可以有多个线程，线程之间是共享地址空间的，同时多核cpu能够让多线程并行执行，成功达到了提高性能的作用。所以也可以理解成，线程解决了进程的并行问题。



> 对于进程和线程，其实还有很多可以讨论的问题，比如进程间的通信方式；多线程需要共享数据，那么如何避免冲突；进程最多可以创建多少个线程；线程崩溃进程也会崩溃吗；死锁，悲观锁，乐观锁，共享锁，排他锁，这么多锁的说法到底是怎么一回事。这些如果讲起来就有点偏离本篇协程的主题了，因此估计会再写一篇文章吧，留个悬念。



### 协程的使用

https://www.cnblogs.com/blizzard8204/p/17563217.html

语言：C++20 C++20的协程是一个无栈，非对称的协程。

```cpp
#include <iostream>
#include <coroutine>
#include <optional>

/*
* promise_type 定义协程的行为，管理协程生命周期中的各种状态。
* Generator 提供协程控制器和迭代器接口，方便协程的使用。
* sequence 是一个协程函数，用来生成一个值序列。
*/

// 协程的返回类型
struct Generator
{
  struct promise_type
  {
    std::optional<int> current_value;

    /* C++编译器通过函数返回值识别协程函数。
    * 返回类型result中有一个子类型承诺对象（promise），
    * 通过std::coroutine_handle<promise_type>::from_promise()
    * 可以得到协程句柄（coroutine handle）。
    */
   // 生成协程函数的返回对象
    Generator get_return_object(){
      return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(int value){
      current_value = value;
      return {};
    }

    void return_void() {}
    void unhandled_exception(){
      std::exit(1); // 未处理异常时退出程序
    }
  };

  // 协程句柄
  std::coroutine_handle<promise_type> handle;

  explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
  ~Generator(){
    if (handle)
      handle.destroy();
  }

  struct Iterator{
    std::coroutine_handle<promise_type> handle;

    Iterator &operator++(){
      handle.resume();
      return *this;
    }

    int operator*() const{
      return *handle.promise().current_value;
    }

    bool operator==(std::default_sentinel_t) const{
      return handle.done();
    }
  };

  Iterator begin(){
    handle.resume();
    return Iterator{handle};
  }

  std::default_sentinel_t end(){
    return {};
  }
};

// 协程函数：生成一个从 1 到 n 的序列
Generator sequence(int n){
  for (int i = 1; i <= n; ++i)
  {
    co_yield i; // 协程暂停并返回值
  }
}

int main(){
  for (int value : sequence(5)){
    std::cout << value << " ";
  }
  return 0;
}
```



### 协程学习

https://lewissbaker.github.io/2017/09/25/coroutine-theory

协程是函数的泛化。

对于函数而言，存在两个操作：call和return。

发生call时，call创建一个激活框架，暂停调用函数的执行，同时将执行转移到调用函数的开始处；return时将返回值传递给调用者，恢复原本调用者的状态，销毁激活框架。这里的激活框架可以视为保存函数特定调用的当前状态的内存块，同时这套激活框架也被成为‘堆栈’。

- 协程中有三个**新的语言关键字**：`co_await`、`co_yield`和`co_return`

- 命名空间中的几种新类型`std::experimental`""
  - `coroutine_handle<P>`
  - `coroutine_traits<Ts...>`
  - `suspend_always`
  - `suspend_never`
- 库编写者可以使用该机制与协程交互并定制其行为。
- 一种语言工具，使编写异步代码变得更加容易！

当前的协程并没有定义协程的语义。它没有定义如何生成返回给调用者的值。它没有定义如何处理传递给语句的返回值`co_return`，或者如何处理从协程传播出去的异常。它没有定义应该在哪个线程上恢复协程。相反，它指定了一种通用机制，让库代码通过实现符合特定接口的类型来定制协程的行为。

比如可以定义一个异步生成单个值的协程，或者一个延迟生成一系列值的协程，或者一个通过遇到值`optional<T>`时提前退出简化使用值的协程。



协程TS定义了两种接口：`Promise`接口和`Awaitable`接口

- **Promise**接口指定了**自定义协程本身行为的方法**。库编写者能够自定义调用协程时发生的情况、协程返回时发生的情况（无论是通过正常方式还是通过未处理的异常），以及自定义协程中任何`co_await`或表达式的行为。`co_yield`
- **Awaitable**接口指定了**控制表达式语义的方法**`co_await` 。当传入一个值时`co_await`，代码将被转换为对 awaitable 对象上的一系列方法的调用，这些方法允许它指定：是否暂停当前协程、在暂停后执行某些逻辑以安排协程稍后恢复、以及在协程恢复后执行某些逻辑以产生表达式的结果`co_await`。



接口清单：

### Awaitable

- awaiter type需要实现如下名字的函数:

1. await_ready
2. await_suspend
3. await_resume

- awaitable type需要实现如下的操作符重载:

1. operator co_await()



### Promise

- promise type需要实现如下名字的函数：

1. get_return_object
2. initial_suspend
3. final_suspend
4. unhandled_exception
5. return_void

- promise type可选实现如下名字的函数：

1. return_value
2. operater new
3. operater delete
4. get_return_object_on_allocation_failure
5. yield_value（co_yield）
6. await_transform



https://yearn.xyz/posts/techs/%E5%8D%8F%E7%A8%8B/

https://itnext.io/c-20-coroutines-complete-guide-7c3fc08db89d