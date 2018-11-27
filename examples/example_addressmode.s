    .function minijava_main 0 1
    call __stdlib_malloc [ $40 ] -> %@0

    movq $42, 16(%@0)
    call __stdlib_println [ 16(%@0) ]

    movq $24, %@1
    movq $23, (%@0,%@1)
    call __stdlib_println [ (%@0, %@1) ]

    addq [ 16(%@0) | (%@0, %@1) ] -> %@2
    call __stdlib_println [ %@2 ]

    movq $0, %@r0
    .endfunction
