#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>

int minijava_main(void);

int64_t __stdlib_read(void)
{
    int result = getchar();
    if (result == EOF) {
        return -1LL;
    } else {
        return result;
    }
}

void __stdlib_println(void)
{
    int64_t value;
    asm("movq 16(%%rbp), %0\n\t" : "=r" (value));
    printf("%" PRIi64 "\n", value);
}

void __stdlib_write(void)
{
    int64_t value;
    asm("movq 16(%%rbp), %0\n\t" : "=r" (value));
    printf("%c", (char)value);
}

void __stdlib_flush(void)
{
    fflush(stdout);
}

void *__stdlib_malloc(void)
{
    int64_t size;
    asm("movq 16(%%rbp), %0\n\t" : "=r" (size));
    return malloc(size);
}

int main(int argc, char **argv)
{
  return minijava_main();
}
