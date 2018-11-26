Molki
=====

molki is an assembly line performing register allocation. It converts input in a pseudo-assembly language which has an unbounded number of registers into x86_64 assembly. The register allocation algorithm is extremely basic, but it works.

Note that this tool has no knowledge about the actual instructions. It merely performs textual replacement of pseudo-registers and certain other patterns.

System Requirements
-------------------

molki requires Python 3.6 or later. In compile and run modes, `gcc` needs to be installed.

The Pseudo-Assembly Language
----------------------------

The pseudo-assembly language is an extension of GNU x86_64 assembly with the following additions:

The directives `.function` and `.endfunction` can be used to generate the prologue and epilogue of functions. `.function` has the following syntax: `.function <name> <number of args> <number of results>`. The maximum number of results is 1.

Within a function defined by `.function`, the pseudo-registers and pseudo-instructions can be used.

Pseudo-registers are referred to as `%@<n>`, where n is a non-negative integer. The first pseudo-registers hold the function's arguments. In addition, the register `%@r0` holds a function's return value.

Pseudo-registers take the same size suffixes as `r8` through `r15`.

There are the following kinds of instructions:

### Basic Instructions

These have the same name and syntax as the x86_64 instructions, only that you can use a pseudo-register where a register would normally be required.

Examples:

```
movq $13, %@42
cmp %@42, %@17
jle somewhere

addl %@30d, (%@31, %@32)
```

### Three Address Instructions

Three address instructions add special syntax to be able to use different registers for the second operand and the result. This syntax is also used for `mul` and `imul`. The syntax is:

```
instr [ <source 1> | <source 2> ] -> <target register>
```

`source 1` and `source 2` may be address mode expressions.

Examples:

```
addq [ %@0 | %@1 ] -> %@2
xorq [ 8(%@6) | (%@7, %@8) ] -> %@9
jg [ %@2 | %@3 ] -> %@42 /* This will result in an invalid instruction */
```

### The `div` and `idiv` instructions

Since `div` and `idiv` produce two results (quotient and remainder), they use yet another syntax:

```
[i]div [ <source 1> | <source 2> ] -> [ <target register div> | <target register mod> ]
```

Example:

```
idiv [ %@0 | %@1 ] -> [ %@2 | %@3 ]
```

### Call instruction

There is a special form of the `call` instruction which handles argument and result passing. Its syntax is:

```
call <function name> [ <argument> | <argument> | ... | <argument> ] (-> <result register>)?
```

There is no check on the validity of the function call.

Example
-------

This code computes the ninth Fibonacci number:

```
.function fib 1 1
    cmpq $1, %@0
    jle fib_basecase
    subq [ $1 | %@0 ] -> %@1
    subq [ $2 | %@0 ] -> %@2
    call fib [ %@1 ] -> %@3
    call fib [ %@2 ] -> %@4
    addq [ %@3 | %@4 ] -> %@r0
    jmp fib_end

fib_basecase:
    movq %@0, %@r0
fib_end:
.endfunction

.function minijava_main 0 1
    movq $9, %@0
    call fib [ %@0 ] -> %@1
    call __stdlib_println [ %@1 ]
.endfunction
```

ABI
---

All arguments are passed on the stack. Every argument occupies eight bytes.

The return value is passed in `%rax`.

Runtime
-------

molki provides an example minijava runtime in `runtime.c`.

License
-------

MIT

Contributions
-------------

Bug fixes and improvements to the pseudo-assembly language are welcome (e.g. support for phi instructions), but please do not try to "improve" the register allocator. It is perfect as-is.

Disclaimer
----------

This is a *really* bad register allocator. For a single `mov %@0, %@1` it produces

```
movq 16(%rbp), %rax
movq -8(%rbp), %rbx
mov %rax, %rbx
movq %rax, 16(%rbp)
movq %rbx, -8(%rbp)
```

You are expected to produce something better than that. You can do it!




Vollendet zu Karlsruhe, den 26. November MXVIII

Johannes Bechberger (noch 020) & Andreas Fried (031)