#!/usr/bin/python3
"""
Basic meta assembler to assembler compiler

Syntax based on GNU assembler (AT&T syntax)

Requires python3.6
"""

import os, re, sys
from enum import Enum
from typing import Dict, List


class MolkiError(Exception):
    pass


class ParseError(MolkiError):
    pass


class RegWidth(Enum):
    """
    Width of a register or instruction.
    Named according to Intel tradition.
    """
    BYTE = 1
    WORD = 2
    DOUBLE = 4
    QUAD = 8


class Register:
    """
    Represents a pseudo-register, represented in source code as %@<index><width-suffix>.
    The index should be numerical or "r0" (the function result pseudo-register),
    the width suffix is assigned in the same way as for r8 through r15:
    'b' for byte, 'w' for word, 'd' for doubleword, and 'q' for quadword.
    """

    def __init__(self, name: str):
        """
        Constructs a new pseudo-register object.
        :param name: The name of the register. Includes the %@ sigill, the index, and the width suffix.
        """
        if not re.match("%@[jr0-9]+[lwd]?", name):
            raise ParseError(f"Register name {name} is invalid.")
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """
        Compares two pseudo-registers by name.
        """
        if isinstance(other, Register):
            return self.name == other.name
        return False

    def __str__(self):
        """
        :return: The name of this register.
        """
        return self.name

    def name_without_width(self) -> str:
        if self.width() != RegWidth.QUAD:
            return self.name[:-1]
        else:
            return self.name

    def width(self) -> RegWidth:
        return {"l": RegWidth.BYTE, "w": RegWidth.WORD, "d": RegWidth.DOUBLE}.get(self.name[-1], RegWidth.QUAD)

class ConcreteRegister:
    """
    Represents a concrete register.
    The name is only the base name of the register, without % or the width suffix.
    The base names are 'a', 'b', 'c', 'd', 'sp', 'bp', 'si', 'di', 'r8' through 'r15'.
    """

    def __init__(self, name: str, width: RegWidth):
        self.name = name
        self.width = width
        # See if the name is valid
        self.asm_name()

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
            return {RegWidth.BYTE: self.name + "b",
                    RegWidth.WORD: self.name + "w",
                    RegWidth.DOUBLE: self.name + "d",
                    RegWidth.QUAD: self.name}[self.width]
        raise ParseError(f"unknown concrete register base name {self.name}")

    def __str__(self):
        """
        :return: The full register name, including % and the width suffix.
        """
        return "%" + self.asm_name()


class RegisterTable:
    """
    Maps register to frame offset (in bytes)
    """

    def __init__(self, func: 'Function'):
        self._raw = {}  # type: Dict[str, int]
        self._cur_offset = 0
        self.function = func

    def __getitem__(self, reg: Register) -> int:
        """
        Returns the frame offset of the given pseudo-register.
        If the register has not yet been seen, a new slot on the frame is allocated,
        assigned to the given register, and returned.
        """
        regname = reg.name_without_width()
        if regname in self._raw:
            return self._raw[regname]
        self._cur_offset -= 8
        self._raw[regname] = self._cur_offset
        return self._cur_offset

    def __setitem__(self, reg: Register, offset: int):
        """
        Manually sets the frame offset of a pseudo-register.
        The offset must be non-negative (i.e. outside the frame),
        and the register must not have an offset yet.
        """
        regname = reg.name_without_width()
        if regname in self._raw:
            raise MolkiError(f"Register {reg} already has offset {self._raw[reg]}")
        if offset <= 0:
            raise MolkiError("Fixed-position registers only allowed at positive offsets")
        self._raw[regname] = offset

    def __str__(self):
        return str(self._raw)

    def size(self) -> int:
        """
        Returns the size of the frame in bytes.
        """
        return -self._cur_offset

    def __repr__(self):
        return repr(self._raw)


class AsmUnit:
    """
    Represents a pseudo-instruction along with its supporting instructions such as movs.
    Instructions are stored in the AsmUnit in a line-based manner.
    The different builder methods each add one or more lines.
    """

    def __init__(self, regs: RegisterTable, usable_regs: List[str] = None):
        """
        :param usable_regs: Base names of the concrete registers which may be used as temporary storage.
        """
        self._lines = []   # type: List[str]
        self._concrete = {}  # type: Dict[Register, str]
        self._regs = regs
        self._usable = usable_regs or ["r8", "r9", "r10", "r11", "r12", "r13"]  # type: List[str]

    def __str__(self):
        """
        :return: The lines accumulated so far.
        """
        return "\n".join(self._lines)

    def raw(self, anything: str) -> 'AsmUnit':
        """
        Appends the given string verbatim as a new line.
        :return: self
        """
        self._lines.append(anything)
        return self

    def comment(self, comment: str) -> 'AsmUnit':
        return self.raw(f"/* {comment.replace('*/', '* /')} */")

    def reserve_register(self, reg: Register) -> 'AsmUnit':
        """
        Reserve the given concrete register for later use.
        """
        conc = self._pop_free_reg()
        self._concrete[reg] = conc
        return self

    def load(self, source: Register) -> 'AsmUnit':
        """
        Add mov from source to any free concrete register.
        """
        conc = self._pop_free_reg()
        self._concrete[source] = conc
        self._lines.append(f"movq {self._regs[source]}(%rbp), {ConcreteRegister(conc, RegWidth.QUAD)}")
        return self

    def loads(self, *sources: Register) -> 'AsmUnit':
        """
        Loads all the given registers.
        """
        for source in sources:
            self.load(source)
        return self

    def instruction(self, line: str) -> 'AsmUnit':
        """
        Add an instruction.
        Replaces pseudo register names with concrete ones added by previous load()/loads()
        """
        result = self._replace_pseudo_regs(line)
        self._lines.append(result)
        return self

    def _replace_pseudo_regs(self, expr: str, reg_width: RegWidth = None) -> str:
        for (reg, conc) in sorted(self._concrete.items(), key=lambda t : len(str(t[0])), reverse=True):
            expr = expr.replace(str(reg), str(ConcreteRegister(conc, reg_width or reg.width())))
        return expr

    def store(self, target: Register) -> 'AsmUnit':
        """
        Adds mov from the concrete register assigned to target to the slot for target on the frame.
        """
        conc = self[target]
        self._lines.append(f"movq {ConcreteRegister(conc, RegWidth.QUAD)}, {self._regs[target]}(%rbp)")
        return self

    def stores(self, *targets: Register) -> 'AsmUnit':
        """
        Stores all the given registers.
        """
        for target in targets:
            self.store(target)
        return self

    def move(self, source: Register, target: Register) -> 'AsmUnit':
        """
        Adds a 64-bit wide move between the concrete registers representing the given pseudo-registers.
        """
        self._lines.append(f"movq {ConcreteRegister(self[source], RegWidth.QUAD)}, {ConcreteRegister(self[target], RegWidth.QUAD)}")
        return self

    def move_to_concrete(self, source: Register, target: str) -> 'AsmUnit':
        """
        Adds a 64-bit wide move from the concrete register representing source
        to the concrete register with base name target.
        """
        self._lines.append(
            f"movq {ConcreteRegister(self[source], RegWidth.QUAD)}, {ConcreteRegister(target, RegWidth.QUAD)}")
        return self

    def move_from_concrete(self, source: str, target: Register) -> 'AsmUnit':
        """
        The inverse of move_to_concrete.
        """
        self._lines.append(
            f"movq {ConcreteRegister(source, RegWidth.QUAD)}, {ConcreteRegister(self[target], RegWidth.QUAD)}")
        return self

    def move_from_anything_to_concrete_reg(self, source: str, target: str) -> 'AsmUnit':
        """
        Adds a 64-bit wide move from source to the concrete register with base name target.
        Source may be any register-, immediate- or address-mode-like expression.
        """
        return self.raw(f"movq {self._replace_pseudo_regs(source, RegWidth.QUAD)}, {ConcreteRegister(target, RegWidth.QUAD)}")

    def _pop_free_reg(self) -> str:
        return self._usable.pop(0)

    def __getitem__(self, reg: Register) -> str:
        """
        :return: The concrete register representing reg.
        """
        return self._concrete[reg]


class Function:
    """
    An assembly function.
    In the source code, a function starts with ".function <name> <number of args> <number of results>".
    The maximum number of results is 1.
    A function ends with ".endfunction".
    The function object takes care of prologue and epilogue.
    """

    def __init__(self, line_number: int, line: str):
        """
        :param line: The line containing the ".function" directive.
        """
        self.line_number = line_number
        self._instrs = []  # type: List[Instruction]
        [keyword, self.name, args, ret] = line.split()
        assert(keyword == ".function")
        self._num_params = int(args)
        self._has_result = ret == "1"

    def extend(self, *instrs: 'Instruction') -> 'Function':
        """
        Add instrs to this function.
        """
        self._instrs.extend(instrs)
        return self

    def toAsm(self) -> str:
        table = RegisterTable(self)
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
                content += "\n"
                if not isinstance(instr, Directive):
                    content += "\n"
            except MolkiError as e:
                print(f"error in line {instr.line_number}: {instr.line}", file=sys.stderr)
                raise e

        result = FUNCTION_HEADER
        result += f"sub ${table.size()}, %rsp\n\n"
        result += content
        result += self.function_return_label() + ":\n"
        if self._has_result:
            result += f"movq {table[Register('%@r0')]}(%rbp), %rax\n"
        result += f"add ${table.size()}, %rsp\n"
        result += FUNCTION_FOOTER

        return result

    def function_return_label(self) -> str:
        return f".{self.name}____________return_block_of_this_function"


def registers_in(raw: str) -> List[Register]:
    """
    Returns all pseudo-registers occurring in raw.
    """
    return list(map(Register, re.findall("%@[jr0-9]+[lwd]?", raw)))


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
        return registers_in(self.line)

    def writeback_registers(self) -> List[Register]:
        """
        :return: A list of pseudo-registers which this instruction writes back to the frame.
           Default is to write back all registers.
        """
        return self.registers()

    @classmethod
    def matches(cls, line: str) -> bool:
        return False

    @classmethod
    def match_line(cls, line_number: int, line: str) -> 'Instruction':
        instruction_constrs = [
            CallInstruction,
            MultInstruction,
            DivInstruction,
            ShiftInstruction,
            ReturnInstruction,
            ThreeAddressCode,
            BasicInstructionNoWriteback,
            BasicInstruction
        ]
        for constr in instruction_constrs:
            if constr.matches(line):
                return constr(line_number, line)

        raise MolkiError(f"No instruction class matches line {line_number}: {line}")


class ThreeAddressCode(Instruction):
    """
    Syntax: instr [ <source 1> | <source 2> ] -> <target register>
    """

    def toAsm(self, regs: RegisterTable) -> str:
        m = re.match(r"([a-z]+)\s*\[([^\]]*)\]\s*->\s*(%@[jr0-9]+[lwd]?)", self.line)
        if not m:
            raise MolkiError("Invalid three-address instruction")

        opcode = m.group(1)
        args = list(map(str.strip, m.group(2).split("|")))
        assert(len(args) == 2)
        source1_raw = args[0]
        source2_raw = args[1]
        target = Register(m.group(3))

        reg_width = target.width()

        actual = self.get_actual_instruction(opcode, reg_width)

        return str(AsmUnit(regs)
                   .comment(self.line)
                   .loads(*self.registers())
                   .move_from_anything_to_concrete_reg(source1_raw, 'd')
                   .move_from_anything_to_concrete_reg(source2_raw, 'a')
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
    Syntax: [i]mul [ <source 1> | <source 2> ] -> <target register>
    """

    @classmethod
    def matches(cls, line: str):
        return ThreeAddressCode.matches(line) and line.startswith("mul") or line.startswith("imul")

    def get_actual_instruction(self, opcode: str, reg_width: RegWidth):
        return f"{opcode} {ConcreteRegister('d', reg_width)}"


class DivInstruction(Instruction):
    """
    Syntax: [i]div [ <source 1> | <source 2> ] -> [ <target register div> | <target register mod> ]
    """

    def toAsm(self, regs: RegisterTable):
        m = re.match(r"([a-z]+)\s*\[([^\]]*)\]\s*->\s*\[([^\]]*)\]", self.line)
        if not m:
            raise MolkiError("Invalid div instruction")

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

        return str(AsmUnit(regs)
                   .comment(self.line)
                   .loads(*self.registers())
                   .move_from_anything_to_concrete_reg(source1_raw, 'a')
                   .move_from_anything_to_concrete_reg(source2_raw, 'b')
                   .raw("cltd")
                   .instruction(f"{opcode} {ConcreteRegister('b', reg_width)}")
                   .move_from_concrete("a", target_div)
                   .store(target_div)) + "\n" + \
               str(AsmUnit(regs)
                   .loads(*self.registers())
                   .move_from_anything_to_concrete_reg(source1_raw, 'a')
                   .move_from_anything_to_concrete_reg(source2_raw, 'b')
                   .raw("cltd")
                   .instruction(f"{opcode} {ConcreteRegister('b', reg_width)}")
                   .move_from_concrete("d", target_mod)
                   .store(target_mod))


    @classmethod
    def matches(cls, line: str):
        return ThreeAddressCode.matches(line) and (line.startswith("div") or line.startswith("idiv"))


class ShiftInstruction(Instruction):
    """
    Syntax: {shl,shr,sal,sar,rol,ror} [ <source 1> | <source 2> ] -> <target register>
    """

    def toAsm(self, regs: RegisterTable):
        m = re.match(r"([a-z]+)\s*\[([^\]]*)\]\s*->\s*(%@[jr0-9]+[lwd]?)", self.line)
        if not m:
            raise MolkiError("Invalid shift instruction")

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

        return str(AsmUnit(regs)
                   .comment(self.line)
                   .loads(source1)
                   .reserve_register(target)
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
    Syntax: call <function name> [ <argument> | <argument> | ... | <argument> ] (-> <result register>)?
    """

    def toAsm(self, regs: RegisterTable):
        m = re.match(r"call\s+([\w]+)\s*\[([^\]]*)\]\s*(->\s*(%@[jr0-9]+[lwd]?))?", self.line)
        if not m:
            raise MolkiError("Invalid call instruction")

        function_name = m.group(1)
        args = list(map(str.strip, m.group(2).split("|"))) if len(m.group(2).strip()) > 0 else []
        result_raw = m.group(4) if len(m.groups()) > 3 else None

        asm_unit = AsmUnit(regs, [])
        asm_unit.comment(self.line)
        for arg in reversed(args):
            pushq = Instruction.match_line(self.line_number, f"pushq {arg}")
            asm_unit.raw(pushq.toAsm(regs))

        asm_unit.raw(f"callq {function_name}")

        if result_raw is not None:
            asm_unit.raw(f"movq %rax, {regs[Register(result_raw)]}(%rbp)")

        for _ in args:
            asm_unit.raw("popq %rbx")

        return str(asm_unit)


    @classmethod
    def matches(cls, line: str):
        return line.startswith("call ") and "[" in line


class BasicInstruction(Instruction):
    """
    Trivially convertable to x86, only pseudo-registers are allocated.
    """

    def toAsm(self, regs: RegisterTable) -> str:
        return str(AsmUnit(regs)
                   .comment(self.line)
                   .loads(*self.registers())
                   .instruction(self.line)
                   .stores(*self.writeback_registers()))

    @classmethod
    def matches(cls, line: str):
        return True


class BasicInstructionNoWriteback(BasicInstruction):

    def writeback_registers(self) -> List[Register]:
        return []

    @classmethod
    def matches(cls, line: str):
        return line.startswith("cmp") or line.startswith("test") or line.startswith("push")


class Directive(Instruction):

    def toAsm(self, _: RegisterTable) -> str:
        return self.line


class ReturnInstruction(Instruction):
    """
    Return instruction, syntax `return` that jumps to the return block
    """

    def toAsm(self, regs: RegisterTable) -> str:
        return f"jmp {regs.function.function_return_label()}"

    @classmethod
    def matches(cls, line: str):
        return line.startswith("return")


def process_lines(lines: List[str]) -> str:
    """
    Converts lines (in pseudo-assembler) to actual assembler.
    """
    pre_lines = []  # type: List[str]
    functions = []  # type: List[Function]
    cur_func = None
    for i, line in enumerate(map(str.strip, lines), start=1):
        if "/*" in line:
            [line, comment] = line.split("/*")
            if cur_func is not None:
                cur_func.extend(Directive(i, "/*" + comment))
            else:
                pre_lines.append("/*" + comment)

        try:
            if len(line.strip()) == 0:
                if cur_func is not None:
                    cur_func.extend(Directive(i, ""))
                else:
                    pre_lines.append("")
            elif line.startswith(".function"):
                if cur_func is not None:
                    print(f"Warning: Inserted missing .endfunction before line {i}", file=sys.stderr)
                cur_func = Function(i, line)
                functions.append(cur_func)
            elif line.startswith(".endfunction"):
                cur_func = None
            elif cur_func is None:
                pre_lines.append(line)
            elif line.startswith("."):
                cur_func.extend(Directive(i, line))
            else:
                cur_func.extend(Instruction.match_line(i, line))
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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="molki is an assembly line performing register allocation written by two nerds once upon a long night in the lab")
    parser.add_argument("mode", metavar="MODE", type=str, choices=["print", "assemble", "compile", "run"],
                        help='What to do with the input program: "print" generates assembly and prints it to stdout with line numbers, ' + \
                             '"assemble" writes assembly to the output file, ' + \
                             '"compile" additionally links in the runtime,' + \
                             '"run" additionally executes the result.')
    parser.add_argument("input", metavar="FILE", type=str, default="-", help="Input file in pseudo-assembly. Defaults to stdin.")
    parser.add_argument("-o", metavar="OUTPUT", type=str, default="a.out", help="Basename for output. Assembly is written to OUTPUT.s, the binary is written to OUTPUT")
    args = parser.parse_args()

    mode = args.mode
    inputfile = "/dev/stdin" if args.input == '-' else args.input
    outputfile = args.o

    with open(inputfile) as f:
        input = f.read()

    result = process(input)

    if mode == "print":
        print("\n".join(f"{i + 1:3}: {line}" for i, line in enumerate(result.splitlines())))
    else:
        # In any other mode, we write the assembly file
        with open(outputfile + ".s", "w") as f:
            f.write(result)
        if mode in ["compile", "run"]:
            script_path = os.path.dirname(__file__)
            os.system(f"gcc {script_path}/runtime.c {outputfile}.s -o {outputfile}")
        if mode == "run":
            os.system(f"./{outputfile}")
