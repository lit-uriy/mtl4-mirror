.globl tuple_call_2
tuple_call_2:
	push %ebp
	mov %esp, %ebp
	push %esi
	push %edi
	push %ecx
	pushf
	cld
	mov 12(%ebp), %esi
	sub 16(%ebp), %esp
	mov %esp, %edi
	mov 16(%ebp), %ecx
	rep movsb
	mov 8(%ebp), %eax
	mov 0(%eax), %eax
	call *%eax
	popf
	pop %ecx
	pop %edi
	pop %esi
	mov %ebp, %esp
	pop %ebp
	ret
