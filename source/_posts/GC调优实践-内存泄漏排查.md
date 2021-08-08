---
title: GC调优实践-内存泄漏排查
date: 2021-07-16 15:05:43
tags: [Java, GC调优]
---

### GC调优实践-内存泄漏排查

看到一篇内存泄漏的文章，动手实践了下。

**场景说明**

使用`HttpAsyncClient`库，访问10000次某一个url，并

设置JVM参数：

```
-XX:+PrintGCDetails -Xloggc:/Users/jiayi/seckill/logs/gc/gcc.log -Xmx200m
```

打印出的日志：

```
229.449: [GC (Allocation Failure) [PSYoungGen: 30720K->2048K(31232K)] 164715K->137644K(167936K), 0.0014919 secs] [Times: user=0.01 sys=0.00, real=0.00 secs] 

246.484: [Full GC (Ergonomics) [PSYoungGen: 28160K->3189K(30720K)] [ParOldGen: 136334K->136334K(136704K)] 164494K->139524K(167424K), [Metaspace: 15467K->15467K(1062912K)], 0.0435582 secs] [Times: user=0.35 sys=0.00, real=0.05 secs] 
```

触发了多次Full GC后，程序最终因`Exception in thread "I/O dispatcher 1" java.lang.OutOfMemoryError: Java heap space`崩溃。

通过VisualVM对程序进行监控：

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gsitq8c3ooj30uo0u0qao.jpg" alt="image-20210716153039219" style="zoom:50%;" />

可以看到，老年代的内存一直在增加。

由以上内容，我们可以猜测是某些对象，因为没有及时释放，导致无法被回收。按照代码来说，我们只涉及到一个`HttpUriRequest`的对象列表，老年代应该不会一直增加。

但是通过查看`HttpClientAsyncService`的内存占用情况，`HttpClientAsyncService$1`一直处于上升状态，肯定是这块代码有了问题。`HttpClientAsyncService$1`表示`HttpClientAsyncService`的内部类，从代码中，我们可以得知是`FutureCallback`匿名内部类。

```java
public List<HttpUriRequest> loadMockRequest(int n){
  List<HttpUriRequest> cache = new ArrayList<>(n);
  for (int i = 0; i < n; i++) {
    cache.add(new HttpGet("http://www.baidu.com?a=" + i));
    // cache.add("http://www.baidu.com?a=" + i);
  }
  return cache;
}
public void start(List<HttpUriRequest> cache) throws InterruptedException {
        final CloseableHttpAsyncClient client = HttpAsyncClients.custom().build();
        client.start();
        int i = 0;
        while (true){
            // String url = cache.get(i%cache.size());
            // final HttpGet request = new HttpGet(url);
            HttpUriRequest request = cache.get(i%cache.size());
            client.execute(request, new FutureCallback<HttpResponse>() {
                @Override
                public void completed(HttpResponse httpResponse) {
                   // System.out.println(request.getRequestLine() + "->" + httpResponse.getStatusLine());
                }

                @Override
                public void failed(Exception e) {
                   // System.out.println(request.getRequestLine() + "->" + e);
                }

                @Override
                public void cancelled() {
                  //  System.out.println(request.getRequestLine() + " cancelled");
                }
            });
            i++;
            Thread.sleep(1000);

        }
    }
```

以上是出现内存泄漏的代码部分，对这块代码进行Debug，我们发现了以下结果：

在Client执行请求时，会将`FutureCallback`构建`Future`对象：

```java
BasicFuture<T> future = new BasicFuture(callback);
```

该对象会参与`DefaultClientExchangeHandlerImpl`的构建：

```java
DefaultClientExchangeHandlerImpl handler = new DefaultClientExchangeHandlerImpl(this.log, requestProducer, responseConsumer, localcontext, future, this.connmgr, this.connReuseStrategy, this.keepaliveStrategy, this.exec);
```

之后，handler调用`start`方法，其中，会将整个handler设置在请求里面

```java
HttpRequest original = this.requestProducer.generateRequest();
((HttpExecutionAware)original).setCancellable(this);
```

这也导致了前面构建的匿名内部类被存在请求里面，因为我们的请求是存在集合当中的，没有办法及时释放，随着Young GC，内部类晋升至Old Gen，最终将老年代空间挤爆。

**解决办法**

使用基本数据类型构建缓存，随时释放请求类，也就释放了内部类。

```java
public List<String> loadMockRequest(int n){
  List<String> cache = new ArrayList<>(n);
  for (int i = 0; i < n; i++) {
    cache.add("http://www.baidu.com?a=" + i);
  }
  return cache;
}
```