---
title: C++初尝
date: 2021-11-10 15:26:09
tags: [C++]
---

## C++初尝

> 阅读笔记《C++ Primer edition 5th》

### chapter 1

```c++
// include，表示要使用某个库或头文件
#include <iostream>
#include "Sales_item.h"

// 标准库命名空间
using namespace std;

int main(){
  // 自定义的类以及与其关联的一组操作
  Sales_item total;
  // cin，标准输入，istream对象，输入流，流的意思是随着时间的推移，字符是顺序生成和消耗的。
  if (cin >> total) {
    Sales_item trans;
    while(cin >> trans) {
      if(total.isbn() == trans.isbn()){
        total += trans;
      } else {
        // cout，标准输出，ostream对象，输出流
        // endl，操纵符的特殊值，效果是结束当前行，将与设备关联的缓冲区内容刷到设备当中
        cout << total << endl;
        total = trans;
      }
    }
    cout << total << endl;
  } else {
    // 输出错误
    cerr << "There is no data!" << endl;
    return -1;
  }
  return 0;
}
// 执行 ./main <data/book_sales >res
```

![image-20211110171045345](https://tva1.sinaimg.cn/large/008i3skNgy1gwa64eem78j313w06omyz.jpg)

这个要如何理解？

首先要了解下**缓冲区**的定义，缓冲区是一块圈定的内存区域，**它用在输入输出设备和CPU之间，用来缓存数据。它使得低速的输入输出设备和高速的CPU能够协调工作，避免低速的输入输出设备占用CPU，解放出CPU，使其能够高效率工作**

**cout**是输出流对象，它会自动关联一块缓冲区，输出的数据一开始是放在缓冲区的，刷新缓冲区操作可以将缓冲区的数据真正地输出到输出设备。

```c++
#include <iostream>
using namespace std;

int main(){
  cout << "abcd"; // 在缓冲区
  cout << endl; // 输出到输出设备
  cout << "def"; // 在缓冲区
}
// 程序结束，刷新缓冲区
```

### chapter2

#### 如何选择正确的数据类型？

1. 当明确数值不可能为负时，选择无符号类型
2. 使用int执行整数运算，如果超出了int的表示范围，选用long long
3. 算术表达式中不要使用char或bool。char类型在某些机器上是有符号的，在另一些机器上又是无符号的
4. 执行浮点运算用double。double精度更高，且与float计算代价相差无几

#### 类型转换

当一个算术表达式中既有无符号数又有int值时，int值会转换为无符号数。

```c++
#include <iostream>

using namespace std;

int main(){
  unsigned u = 10, u2 = 42;
  cout << u2 - u << endl;
  cout << u - u2 << endl; // 4294967264, 有符号数相减结果为-32，因为是无符号表示，所以实际值为2^32 - 32（补码表示法）

  int i = 10, i2 = 42;
  cout << i2 - i << endl;
  cout << i - i2 << endl;


  cout << i - u << endl;
  cout << u - i << endl;

  int i3 = -10;
  cout << i3 - u << endl; // 4294967276
  cout << u + i3 << endl;

  int i4 = -20;
  cout << u + i4 << endl; // 4294967286
  cout << i4 + u << endl; // 4294967286
}
```

#### 花括号初始化（列表初始化）

列表初始化在C++11后，可以支持包括基本类型在内的所有类型的初始化。

```c++
#include <iostream>

using namespace std;
class Example {
  public:
    int a;
    int b;
    Example() = default;
    Example(int aIn, int bIn) {
      a = aIn;
      b = bIn;
    };
};
int main() {
  string str("hello");
  string str2 = "world";
  string str3 = {"!"};

  cout << str + str2 + str3 << endl;

  int a = 0.1; // warning 级别，潜在的类型转换，不会报编译错误
  // int b = {0.1}; // error级别，报编译错误
  
  // class type也可以用列表初始化
  Example example = {1,1};
  cout << example.a << example.b << endl;
}
```

#### 变量命名

C++支持**分离式编译**。

```c++
extern int i; // 声明变量i，但未进行定义，主要用于文件间代码共享
int j; // 声明并定义
extern int k = 3; // 定义
```

#### 作用域

C++允许在内层作用域中重新定义外层作用域中已有的名字，将其覆盖掉。

```c++
int i = 100;
for (int i = 0; i < 10; i++)
{
  cout<<i;
}
```

#### 复合类型

*复合类型指基于其它类型定义的类型*

##### 指针和引用的区别

指针的定义：

Pointers store address of variables or a memory location. 

指针存储变量的地址或指向一块内存地址。

```c++
  int var = 10;
  int *p = &var;
  cout << *p << endl; // 10
  cout << p << endl; // 0x7ffa0...

  *p = 20;
  cout << var << endl; // 20
```

![image-20211121165847840](https://tva1.sinaimg.cn/large/008i3skNgy1gwmvlekx59j30xo0gwmya.jpg)

指针对象支持自增（++）与自减（--）操作，增加的长度为指针类型的长度，如int指针增加4

![image-20211121170353544](/Users/jiayi/Library/Application%20Support/typora-user-images/image-20211121170353544.png)

**引用**的定义：

When a variable is declared as a reference, it becomes an alternative name for an existing variable.

引用其实就是给已存在的对象起了一个别名，类似于“阿里的花名”😁，给风清扬一拳，也表示给了马云一拳。

**常用场景**：

1. **Modify the passed parameters in a function**，修改函数传参

   ```c++
     int a = 1, b = 2;
     swap(a,b);
     cout << a << " " << b << endl;
     void swap(int &a, int &b){
       int tmp = a;
       a = b;
       b = a;
     }
   ```

2. **Avoiding a** **copy of large structures**，避免大对象的copy浪费空间

   在函数传递时，通过引用类型，避免大对象的copy

3. **In For Each Loops to modify all objects**

   在foreach中，修改对象

   ```c++
   vector<int> vect{ 10, 20, 30, 40 }; 
   for(int &x:vect) {
     x = x + 5;
   }
   for(int x:vect){
     cout << x; // {15, 25, 35, 45}
   }
   ```

4. **For Each Loop to avoid the** **copy of objects**

   ```c++
   vector<string> strs{"aaabbb", "bbbccc", "cccddd"};
   for(string &str: strs){
     cout << str; // 避免进行值copy
   }
   ```

```c++
 int k, &rk = k; // & 表示rk是一个reference，引用型变量，rk可以理解为k的别名
  k = 5; rk = 10; //  对rk进行操作，会影响到k
  cout << k << " " << rk << endl;

  int i = 10;
  int *p = nullptr; // 空指针，*表示p为指针变量
  p = &i; // 给指针变量赋值，&为取地址符号
  cout << *p << endl; // *为解引用符，对指针解引用可以得到所指的对象，因此可以获得i
```

##### Pointer vs Reference

1. 引用初始化**必须进行赋值**，且**不可变**，指针则本身是一个对象，允许对它进行赋值和拷贝，**可以先后指向不同的对象**，定义时也**不须赋初值**
2. 在调用栈上，**指针有自己的地址和内存空间**，而引用与原变量共享同一个地址。

```c++
int test  = 10, *pt = &test, &rt = test;
cout << &test << " " << &pt << " " << &rt << " " << pt << endl;
// 0x7ffeeca40454 0x7ffeeca40448 0x7ffeeca40454 0x7ffeeca40454
```

1. 指针可以有多级，而引用只提供一级

```c++
int **p; // valid
int &&q;// error
```





