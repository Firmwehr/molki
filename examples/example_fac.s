.function fac 1 1
    movq $1, %@1
fac_header:
    cmpq $1, %@0
    jle fac_end
    mulq [ %@0 | %@1 ] -> %@1
    subq [ $1 | %@0 ] -> %@0
    jmp fac_header

fac_end:
    movq %@1, %@r0
.endfunction

.function minijava_main 0 1
    movq $4, %@0
    call fac [ %@0 ] -> %@1
    call __stdlib_println [ %@1 ]
.endfunction
