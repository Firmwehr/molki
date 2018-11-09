#!/usr/bin/python3
"""
Basic meta assembler to assembler compiler

Syntax based on GNU assembler (AT&T syntax)

Requires python3.6


TODO
- cli frontend: ./script input output
- language specification
- testing

"""

import os
import tempfile
from enum import Enum
from typing import Dict, Iterable, List
import re

import sys


class MolkiError(Exception):
    pass

class ParseError(MolkiError):
    pass

class RegWidth(Enum):
    BYTE = 1
    WORD = 2
    DOUBLE = 4
    QUAD = 8

class Register:

    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Register):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def width(self) -> RegWidth:
        return {"l": RegWidth.BYTE, "w": RegWidth.WORD, "d": RegWidth.DOUBLE}.get(self.name[-1], RegWidth.QUAD)

class ConcreteRegister:

    def __init__(self, name: str, width: RegWidth):
        self.name = name
        self.width = width

    def asm_name(self) -> str:
        if self.name in ["a", "b", "c", "d"]:
            return {RegWidth.BYTE: self.name + "l",
                    RegWidth.WORD: self.name + "x",
                    RegWidth.DOUBLE: "e" + self.name + "x",
                    RegWidth.QUAD: "r" + self.name + "x"}[self.width]
        elif self.name in ["sp", "bp", "si", "di"]:
            return {RegWidth.BYTE: self.name + "l",
                    RegWidth.WORD: self.name,
                    RegWidth.DOUBLE: "e" + self.name,
                    RegWidth.QUAD: "r" + self.name}[self.width]
        elif self.name in ["r" + str(i) for i in range(8, 16)]:
            return {RegWidth.BYTE: self.name + "l",
                    RegWidth.WORD: self.name + "w",
                    RegWidth.DOUBLE: self.name + "d",
                    RegWidth.QUAD: self.name}[self.width]
        raise MolkiError(f"unknown register {self.name}")

    def __str__(self):
        return "%" + self.asm_name()


class RegisterTable:
    """
    Maps register to offset (in bytes)
    """

    def __init__(self):
        self._raw = {}  # type: Dict[Register, int]
        self._cur_offset = 0

    def __getitem__(self, reg: Register) -> int:
        if reg in self._raw:
            return self._raw[reg]
        self._cur_offset -= 8
        self._raw[reg] = self._cur_offset
        return self._cur_offset

    def __setitem__(self, reg: Register, offset: int):
        if reg in self._raw:
            raise MolkiError(f"Register {reg} already has offset {self._raw[reg]}")
        if offset <= 0:
            raise MolkiError("Fixed-position registers only allowed at positive offsets")
        self._raw[reg] = offset

    def __str__(self):
        return str(self._raw)

    def size(self) -> int:
        return -self._cur_offset

    def __repr__(self):
        return repr(self._raw)


class AsmUnit:

    def __init__(self, regs: RegisterTable, usable_regs: List[str] = None):
        self._lines = []   # type: List[str]
        self._concrete = {}  # type: Dict[Register, str]
        self._regs = regs
        self._usable = usable_regs or ["a", "b", "c", "d"]  # type: List[str]

    def __str__(self):
        return "\n".join(self._lines)

    def raw(self, anything: str) -> 'AsmUnit':
        self._lines.append(anything)
        return self

    def comment(self, comment: str) -> 'AsmUnit':
        return self.raw(f"/* {comment} */")

    def allocate_register(self, reg: Register) -> 'AsmUnit':
        """
        Add mov from source to any free register
        """
        conc = self._pop_free_reg()
        self._concrete[reg] = conc
        return self

    def load(self, source: Register) -> 'AsmUnit':
        """
        Add mov from source to any free register
        """
        conc = self._pop_free_reg()
        self._concrete[source] = conc
        self._lines.append(f"movq {self._regs[source]}(%rbp), {ConcreteRegister(conc, RegWidth.QUAD)}")
        return self

    def loads(self, *sources: Register) -> 'AsmUnit':
        """
        Add mov from source to any free register
        """
        for source in sources:
            self.load(source)
        return self

    def instruction(self, line: str) -> 'AsmUnit':
        """
        Add an instruction. Replaces pseudo register names with concrete ones constructed by previous load()/loads()
        """
        result = self.replace_pseudo_regs(line)
        self._lines.append(result)
        return self

    def replace_pseudo_regs(self, expr: str) -> str:
        for (reg, conc) in self._concrete.items():
            expr = expr.replace(str(reg), str(ConcreteRegister(conc, reg.width())))
        return expr

    def store(self, target: Register) -> 'AsmUnit':
        """
        Adds mov…
        """
        conc = self[target]
        self._lines.append(f"movq {ConcreteRegister(conc, RegWidth.QUAD)}, {self._regs[target]}(%rbp)")
        return self

    def move(self, source: Register, target: Register) -> 'AsmUnit':
        self._lines.append(f"movq {ConcreteRegister(self[source], RegWidth.QUAD)}, {ConcreteRegister(self[target], RegWidth.QUAD)}")
        return self

    def move_to_concrete(self, source: Register, target: str) -> 'AsmUnit':
        self._lines.append(
            f"movq {ConcreteRegister(self[source], RegWidth.QUAD)}, {ConcreteRegister(target, RegWidth.QUAD)}")
        return self

    def move_from_concrete(self, source: str, target: Register) -> 'AsmUnit':
        self._lines.append(
            f"movq {ConcreteRegister(source, RegWidth.QUAD)}, {ConcreteRegister(self[target], RegWidth.QUAD)}")
        return self

    def move_from_anything_to_concrete_reg(self, source: str, target: ConcreteRegister) -> 'AsmUnit':
        return self.raw(f"movq {self.replace_pseudo_regs(source)}, {target}")

    def stores(self, *targets: Register) -> 'AsmUnit':
        for target in targets:
            self.store(target)
        return self

    def _pop_free_reg(self) -> str:
        return self._usable.pop(0)

    def __getitem__(self, reg: Register) -> str:
        return self._concrete[reg]


class Function:

    def __init__(self, line_number: int, line: str):
        self.line_number = line_number
        self._instrs = []  # type: List[Instruction]
        [keyword, self.name, args, ret] = line.split()
        assert(keyword == ".function")
        self._num_params = int(args)
        self._has_result = ret == "1"

    def extend(self, *instrs: 'Instruction') -> 'Function':
        self._instrs.extend(instrs)
        return self

    def toAsm(self) -> str:
        table = RegisterTable()
        for i in range(self._num_params):
            reg = Register(f"%@{i}")
            table[reg] = 16 + 8 * i

        _ = table[Register("%@r0")] # side-effect is to reserve a stack slot

        FUNCTION_HEADER  = f""".globl	{self.name}
.type	{self.name}, @function

{self.name}:
pushq	%rbp
movq	%rsp, %rbp

"""
        FUNCTION_FOOTER  = """popq %rbp
ret
"""

        content = ""
        for instr in self._instrs:
            try:
                content += instr.toAsm(table)
                content += "\n\n"
            except MolkiError as e:
                print(f"error in line {instr.line_number}: {instr.line}", file=sys.stderr)
                raise e

        result = FUNCTION_HEADER
        result += f"sub ${table.size()}, %rsp\n\n"
        result += content
        if self._has_result:
            result += f"movq {table[Register('%@r0')]}(%rbp), %rax\n"
        else:
            result += f"movq $0xBAAAAAAAAAADF00D, %rax\n"
        result += f"add ${table.size()}, %rsp\n"
        result += FUNCTION_FOOTER

        return result

class Instruction:
    """
    Might be anything, even a label: everything that constitutes a line in assembler code
    """

    def __init__(self, line_number: int, line: str):
        self.line_number = line_number
        self.line = line
        """ assembler source code line """

    def toAsm(self, regs: RegisterTable) -> str:
        raise NotImplementedError()

    def registers(self) -> List[Register]:
        return list(map(Register, re.findall("%@[jr0-9]+[lwd]?", self.line)))

    @classmethod
    def matches(cls, line: str) -> bool:
        return False

class SpecialInstruction(Instruction):
    """
    Need extra handling, but have direct counter parts in x86
    """


class ThreeAddressCode(Instruction):
    """
    instr [ <source 1> | <source 2> ] -> <target register>
    """

    def toAsm(self, regs: RegisterTable) -> str:
        m = re.match(r"([a-z]+)\s*\[([^\]]*)\]\s*->\s*(%@[jr0-9]+[lwd]?)", self.line)
        if not m:
            raise MolkiError("Andi fails at regex")

        opcode = m.group(1)
        args = list(map(str.strip, m.group(2).split("|")))
        assert(len(args) == 2)
        source1_raw = args[0]
        source2_raw = args[1]
        target = Register(m.group(3))

        reg_width = target.width()

        actual = self.get_actual_instruction(opcode, reg_width)

        return str(AsmUnit(regs, ["b", "c", "si", "di", "r8"])
                   .comment(self.line)
                   .loads(*self.registers())
                   .move_from_anything_to_concrete_reg(source1_raw, ConcreteRegister('d', RegWidth.QUAD))
                   .move_from_anything_to_concrete_reg(source2_raw, ConcreteRegister('a', RegWidth.QUAD))
                   .instruction(actual)
                   .move_from_concrete("a", target)
                   .store(target))

    @classmethod
    def matches(cls, line: str) -> bool:
        return " -> " in line

    def get_actual_instruction(self, opcode: str, reg_width: RegWidth):
        return f"{opcode} {ConcreteRegister('d', reg_width)}, {ConcreteRegister('a', reg_width)}"


class MultInstruction(ThreeAddressCode):
    """
    [i]mul [ <source 1> | <source 2> ] -> <target register>
    """

    @classmethod
    def matches(cls, line: str):
        return ThreeAddressCode.matches(line) and line.startswith("mul") or line.startswith("imul")

    def get_actual_instruction(self, opcode: str, reg_width: RegWidth):
        return f"{opcode} {ConcreteRegister('d', reg_width)}"


class DivInstruction(Instruction):
    """
    [i]div [ <source 1> | <source 2> ] -> [ <target register div> | <target register mod> ]
    """

    def toAsm(self, regs: RegisterTable):
        m = re.match(r"([a-z]+)\s*\[([^\]]*)\]\s*->\s*\[([^\]]*)\]", self.line)
        if not m:
            raise MolkiError("Andi fails at regex")

        opcode = m.group(1)
        args = list(map(str.strip, m.group(2).split("|")))
        targets = list(map(str.strip, m.group(3).split("|")))
        assert(len(args) == 2)
        source1_raw = args[0]
        source2_raw = args[1]
        assert(len(targets) == 2)
        target_div = Register(targets[0])
        target_mod = Register(targets[1])

        reg_width = target_div.width()

        return str(AsmUnit(regs, ["c", "si", "di", "r8", "r9", "r10"])
                   .comment(self.line)
                   .loads(*self.registers())
                   .move_from_anything_to_concrete_reg(source1_raw, ConcreteRegister('a', RegWidth.QUAD))
                   .move_from_anything_to_concrete_reg(source2_raw, ConcreteRegister('b', RegWidth.QUAD))
                   .raw("cltd")
                   .instruction(f"{opcode} {ConcreteRegister('b', reg_width)}")
                   .move_from_concrete("a", target_div)
                   .store(target_div)) + "\n" + \
               str(AsmUnit(regs, ["c", "si", "di", "r8", "r9", "r10"])
                   .loads(*self.registers())
                   .move_from_anything_to_concrete_reg(source1_raw, ConcreteRegister('a', RegWidth.QUAD))
                   .move_from_anything_to_concrete_reg(source2_raw, ConcreteRegister('b', RegWidth.QUAD))
                   .raw("cltd")
                   .instruction(f"{opcode} {ConcreteRegister('b', reg_width)}")
                   .move_from_concrete("d", target_mod)
                   .store(target_mod))


    @classmethod
    def matches(cls, line: str):
        return ThreeAddressCode.matches(line) and line.startswith("div") or line.startswith("idiv")


class ShiftInstruction(Instruction):
    """
    {shl,shr,sal,sar,rol,ror} [ <source 1> | <source 2> ] -> <target register>
    """

    def toAsm(self, regs: RegisterTable):
        m = re.match(r"([a-z]+)\s*\[([^\]]*)\]\s*->\s*(%@[jr0-9]+[lwd]?)", self.line)
        if not m:
            raise MolkiError("Andi fails at regex")

        opcode = m.group(1)
        args = list(map(str.strip, m.group(2).split("|")))
        assert(len(args) == 2)
        source1_raw = args[0]
        source2_raw = args[1]
        target = Register(m.group(3))

        source1 = Register(source1_raw)
        if "%" in source2_raw:
            source2 = f"{regs[Register(source2_raw)]}(%rbp)"
        else:
            source2 = source2_raw

        instr = f"{opcode} %cl, {source1_raw}"

        return str(AsmUnit(regs, ["a", "b", "d"])
                   .comment(self.line)
                   .loads(source1)
                   .allocate_register(target)
                   .raw(f"movb {source2}, %cl")
                   .instruction(instr)
                   .move(source1, target)
                   .store(target))

    @classmethod
    def matches(cls, line: str):
        return ThreeAddressCode.matches(line) and \
               line[0:3] in ["shl", "shr", "sal", "sar", "rol", "ror"]

class MetaInstruction(Instruction):
    """
    Do not have a counter part in x86
    """

class CallInstruction(MetaInstruction):
    """
    call <function name> [ <argument register or immediate> | <argument> | ... | <argument> ] (-> <result register>)?
    """

    def toAsm(self, regs: RegisterTable):
        m = re.match(r"call\s+([\w]+)\s*\[([^\]]*)\]\s*(->\s*(%@[jr0-9]+[lwd]?))?", self.line)
        if not m:
            raise MolkiError("Andi fails at regex")

        function_name = m.group(1)
        args = list(map(str.strip, m.group(2).split("|")))
        result_raw = m.group(4) if len(m.groups()) > 3 else None

        asm_unit = AsmUnit(regs, [])
        asm_unit.comment(self.line)
        for arg in args:
            arg_source = None
            if "%" in arg:
                arg_source = f"{regs[Register(arg)]}(%rbp)"
            else:
                arg_source = arg
            asm_unit.raw(f"pushq {arg_source}")

        asm_unit.raw(f"callq {function_name}")

        if result_raw is not None:
            asm_unit.raw(f"movq %rax, {regs[Register(result_raw)]}(%rbp)")

        for _ in args:
            asm_unit.raw("popq %rbx")

        return str(asm_unit)


    @classmethod
    def matches(cls, line: str):
        return line.startswith("call ")

class BasicInstruction(Instruction):
    """
    Trivially convertable to x86
    """

    def toAsm(self, regs: RegisterTable) -> str:
        return str(AsmUnit(regs)
                   .comment(self.line)
                   .loads(*self.registers())
                   .instruction(self.line)
                   .stores(*self.registers()))

    @classmethod
    def matches(cls, line: str):
        return True


class Directive(Instruction):

    def toAsm(self, _: RegisterTable) -> str:
        return self.line


def process_lines(lines: List[str]) -> str:
    pre_lines = []  # type: List[str]
    functions = []  # type: List[Function]
    cur_func = None
    for i, line in enumerate(map(str.strip, lines), start=1):
        try:
            if line.startswith(".function"):
                cur_func = Function(i, line)
                functions.append(cur_func)
            elif cur_func is None:
                pre_lines.append(line)
            elif line.startswith("."):
                cur_func.extend(Directive(i, line))
            else:
                instr = None
                instruction_constrs = [
                    CallInstruction,
                    MultInstruction,
                    DivInstruction,
                    ShiftInstruction,
                    ThreeAddressCode,
                    BasicInstruction
                ]
                for constr in instruction_constrs:
                    if constr.matches(line):
                        cur_func.extend(constr(i, line))
                        break
        except Exception as e:
            print(f"error in line {i}", file=sys.stderr)
            raise e

    result = "\n".join(pre_lines) + """
    .text
    """
    for f in functions:
        result += "\n\n"
        result += f.toAsm()
    return result


def process(code: str) -> str:
    return process_lines(code.splitlines())


def assemble(file: str, output: str = "test.o"):
    print("\n".join(f"{i + 1}: {line}" for i, line in enumerate(file.splitlines())))
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w") as f:
        print(file, file=f)
        f.flush()
        os.system(f"as {f.name} -o {output}")


def compile_and_run(file: str, output: str = "test"):
    assemble(file, output + ".o")
    print("return code: " + str(os.system(f"gcc runtime.c {output}.o -o {output} && ./{output}") >> 8))

if False:
    compile_and_run(process("""
    .function minijava_main 0 1
    movq $21, %@0
    addq [ %@0 | %@0 ] -> %@1
    movq %@1, %@r0
    """))

if False:
    compile_and_run(process("""
    .function minijava_main 0 1
    movq $5, %@0
    movq $2, %@1
    subq [ %@1 | %@0 ] -> %@2
    call __stdlib_println [ %@2 ]
    subq [ %@0 | %@1 ] -> %@3
    call __stdlib_println [ %@3 ]
    """))

if False:
    compile_and_run(process("""
    .function minijava_main 0 1
    movq $13, %@0
    movq $5, %@1
    mulq [ %@0 | %@1 ] -> %@2
    movq %@2, %@r0
    """))

if False:
    compile_and_run(process("""
    .function minijava_main 0 1
    movq $13, %@0
    movq $5, %@1
    divq [ %@0 | %@1 ] -> [%@2 | %@3]
    call __stdlib_println [ %@2 ]
    call __stdlib_println [ %@3 ]
    """))

if True:
    compile_and_run(process("""
    .function minijava_main 0 1
    movq $1, %@0
    movq $7, %@1
    shl [%@0| %@1] -> %@2
    movq %@2, %@r0
    """))

if False:
    compile_and_run(process("""
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

    .function minijava_main 0 1
    movq $9, %@0
    call fib [ %@0 ] -> %@r0
    """))

if False:
    compile_and_run(process("""
    .function minijava_main 0 1
    call __stdlib_println [ $42424242 ]
    """))
