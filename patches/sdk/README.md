# Platform patches

A patch to the title heap in PS5_PayloadSDK's platform layer (`platform/src/heap.c`, dlmalloc
as eight per-thread arenas in direct memory), applied on top of the revision PS5_Vulkan pins
(`b83202b`):

- `0001`: the heap gives memory back. Arenas grow 2 MiB at a time instead of 16 MiB (as
  `9d04878` does upstream), a block from 1 MiB has a segment of its own that is returned when
  it is freed (the threshold was 32 MiB), frees look for wholly free segments every 255 calls,
  and `ps5_heap_trim()` returns every arena's free segments at once. The app trims after each
  request (`ps5/app/activity.cpp`).

Without it, llama-server held each arena at the largest request it had served, since each
request runs on another HTTP thread. With a chat whose history grows to 48k tokens the pool's
last half GiB filled 16 MiB at a time and the model stopped. On a macOS host with the kernel
calls stubbed, a request pattern on 15 threads held 640 MiB with the old settings and 54 MiB
with the patch and a trim after each request. Not yet measured on the console.

`scripts/build-app.sh` applies it by itself: it exports the platform layer from the fork at
the SDK's revision (`.ps5-sdk-revision`), patches it, builds `libps5platform.a` and links that
in place of the SDK's. The SDK as installed is not changed. To patch the fork instead:

```sh
cd .deps/src/PS5_PayloadSDK && git am ../../../patches/sdk/*.patch
```
