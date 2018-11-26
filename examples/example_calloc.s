	.function minijava_main 0 1

	movq $4, %@0
	movq $8, %@1

	call __stdlib_calloc [%@0 | %@1] -> %@2

	call __stdlib_println [ 16(%@2) ]

	movq $0, %@0
