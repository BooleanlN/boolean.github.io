---
title: Redis-内存分配器
date: 2021-11-28 19:54:54
tags: [NoSQL, Redis]
---

## 内存分配器

分析下zmalloc.h、zmalloc.c两个文件源码。

### Summary

### Detail

1. 内存分配器

   ```c
   #if defined(USE_TCMALLOC)
   # do something with tcmalloc
   # endif
   
   #elif defined(USE_JEMALLOC)
   # do something with jemalloc
   #endif
   
   #elif defined(__APPLE__)
   # do something on Apple platform
   #endif
   
   ```

   Redis提供了三种内存分配器选择，包括tcmalloc、jemalloc、Apple。

   

```c++
void *zmalloc(size_t size);
void *zcalloc(size_t size);
void *zrealloc(void *ptr, size_t size);
void *ztrymalloc(size_t size);
void *ztrycalloc(size_t size);
void *ztryrealloc(void *ptr, size_t size);
void zfree(void *ptr);
void *zmalloc_usable(size_t size, size_t *usable);
void *zcalloc_usable(size_t size, size_t *usable);
void *zrealloc_usable(void *ptr, size_t size, size_t *usable);
void *ztrymalloc_usable(size_t size, size_t *usable);
void *ztrycalloc_usable(size_t size, size_t *usable);
void *ztryrealloc_usable(void *ptr, size_t size, size_t *usable);
void zfree_usable(void *ptr, size_t *usable);
char *zstrdup(const char *s);
size_t zmalloc_used_memory(void);
void zmalloc_set_oom_handler(void (*oom_handler)(size_t));
size_t zmalloc_get_rss(void);
int zmalloc_get_allocator_info(size_t *allocated, size_t *active, size_t *resident);
void set_jemalloc_bg_thread(int enable);
int jemalloc_purge();
size_t zmalloc_get_private_dirty(long pid);
size_t zmalloc_get_smap_bytes_by_field(char *field, long pid);
size_t zmalloc_get_memory_size(void);
void zlibc_free(void *ptr);
```



