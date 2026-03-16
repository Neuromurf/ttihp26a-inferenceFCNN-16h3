"""
test.py - Cocotb testbench for tt_um_neuromurf_seq_mac_inf
Compatible with RTL and gate-level simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import os

# Gate-level needs longer waits
GL_TEST = os.environ.get("GATES", "no") == "yes"
TIMING_MULT = 3 if GL_TEST else 1

CLK_PERIOD_NS = 20
MODE_IDLE = 0b00
MODE_MAC_ONLY = 0b01
MODE_INFERENCE = 0b10

BIT_TOGGLE = 0
BIT_TYPE = 1
BIT_START = 2
BIT_MODE0 = 3
BIT_NEXT = 4
BIT_SOFT_RST = 5
BIT_MODE1 = 6
BIT_STATUS = 7


class Driver:
    def __init__(self, dut):
        self.dut = dut
        self.toggle = 0
        self.next_toggle = 1

    def _uio(self, toggle=None, dtype=0, start=0, mode=0, next_byte=None, soft_rst=0, status_sel=0):
        if toggle is None:
            toggle = self.toggle
        if next_byte is None:
            next_byte = self.next_toggle
        return (
            (toggle << BIT_TOGGLE) |
            (dtype << BIT_TYPE) |
            (start << BIT_START) |
            ((mode & 1) << BIT_MODE0) |
            (next_byte << BIT_NEXT) |
            (soft_rst << BIT_SOFT_RST) |
            ((mode >> 1) << BIT_MODE1) |
            (status_sel << BIT_STATUS)
        )

    async def reset(self):
        self.dut.ena.value = 1
        self.dut.ui_in.value = 0
        self.dut.uio_in.value = 0
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, 10 * TIMING_MULT)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 10 * TIMING_MULT)
        self.toggle = 0
        self.next_toggle = 1

    async def status(self):
        self.dut.uio_in.value = self._uio(status_sel=1)
        await ClockCycles(self.dut.clk, 2 * TIMING_MULT)
        return int(self.dut.uo_out.value)

    async def data(self):
        self.dut.uio_in.value = self._uio(status_sel=0)
        await ClockCycles(self.dut.clk, 2 * TIMING_MULT)
        return int(self.dut.uo_out.value)

    async def wait_ready(self, timeout=200):
        for _ in range(timeout * TIMING_MULT):
            if (await self.status()) & 0x04:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False

    async def wait_byte_valid(self, timeout=200):
        for _ in range(timeout * TIMING_MULT):
            if (await self.status()) & 0x08:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False

    async def wait_inf_done(self, timeout=500):
        for _ in range(timeout * TIMING_MULT):
            if (await self.status()) & 0x10:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False

    async def start_mode(self, mode):
        self.dut.uio_in.value = self._uio(start=1, mode=mode)
        await ClockCycles(self.dut.clk, 5 * TIMING_MULT)
        self.dut.uio_in.value = self._uio(start=0, mode=mode)
        await ClockCycles(self.dut.clk, 5 * TIMING_MULT)

    async def send_weight(self, weight, mode):
        self.dut.ui_in.value = weight & 0xFF
        self.dut.uio_in.value = self._uio(dtype=0, mode=mode)
        await ClockCycles(self.dut.clk, 2 * TIMING_MULT)
        self.toggle = 1 - self.toggle
        self.dut.uio_in.value = self._uio(dtype=0, mode=mode)
        await ClockCycles(self.dut.clk, 2 * TIMING_MULT)

    async def send_input(self, inp, mode):
        self.dut.ui_in.value = inp & 0xFF
        self.dut.uio_in.value = self._uio(dtype=1, mode=mode)
        await ClockCycles(self.dut.clk, 2 * TIMING_MULT)
        self.toggle = 1 - self.toggle
        self.dut.uio_in.value = self._uio(dtype=1, mode=mode)
        await ClockCycles(self.dut.clk, 2 * TIMING_MULT)

    async def send_pair(self, weight, inp, mode):
        assert await self.wait_ready(), "Timeout waiting for ready (weight)"
        await self.send_weight(weight, mode)
        assert await self.wait_ready(), "Timeout waiting for ready (input)"
        await self.send_input(inp, mode)
        await ClockCycles(self.dut.clk, 20 * TIMING_MULT)

    async def next_byte(self, mode):
        self.next_toggle = 1 - self.next_toggle
        self.dut.uio_in.value = self._uio(mode=mode, status_sel=0)
        await ClockCycles(self.dut.clk, 15 * TIMING_MULT)


@cocotb.test()
async def test_reset(dut):
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    d = Driver(dut)
    await d.reset()
    s = await d.status()
    assert (s & 0x01) == 0, f"busy should be 0"
    assert (s & 0x40) == 0, f"err_flag should be 0"
    dut._log.info("PASS: test_reset")


@cocotb.test()
async def test_mac_simple(dut):
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    d = Driver(dut)
    await d.reset()
    await d.start_mode(MODE_MAC_ONLY)
    for _ in range(9):
        await d.send_pair(1, 1, MODE_MAC_ONLY)
    await d.wait_byte_valid(100)
    result = await d.data()
    dut._log.info(f"MAC result: {result}, expected: 9")
    assert result == 9, f"Expected 9, got {result}"
    dut._log.info("PASS: test_mac_simple")


@cocotb.test()
async def test_mac_larger(dut):
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    d = Driver(dut)
    await d.reset()
    await d.start_mode(MODE_MAC_ONLY)
    for _ in range(9):
        await d.send_pair(10, 20, MODE_MAC_ONLY)
    await d.wait_byte_valid(100)
    byte0 = await d.data()
    await d.next_byte(MODE_MAC_ONLY)
    byte1 = await d.data()
    await d.next_byte(MODE_MAC_ONLY)
    byte2 = await d.data()
    result = byte0 | (byte1 << 8) | (byte2 << 16)
    dut._log.info(f"MAC result: {result} (bytes: {byte0}, {byte1}, {byte2}), expected: 1800")
    assert result == 1800, f"Expected 1800, got {result}"
    dut._log.info("PASS: test_mac_larger")


@cocotb.test()
async def test_soft_reset(dut):
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    d = Driver(dut)
    await d.reset()
    await d.start_mode(MODE_MAC_ONLY)
    s = await d.status()
    assert s & 0x01, "Should be busy"
    d.dut.uio_in.value = d._uio(soft_rst=1)
    await ClockCycles(dut.clk, 10 * TIMING_MULT)
    d.dut.uio_in.value = d._uio(soft_rst=0)
    await ClockCycles(dut.clk, 10 * TIMING_MULT)
    s = await d.status()
    assert (s & 0x01) == 0, f"busy should be 0"
    dut._log.info("PASS: test_soft_reset")


@cocotb.test()
async def test_inference(dut):
    """Skip for gate-level (too slow)."""
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    if GL_TEST:
        dut._log.info("SKIP: Inference test disabled for gate-level")
        return
    d = Driver(dut)
    await d.reset()
    await d.start_mode(MODE_INFERENCE)
    dut._log.info("L1: 16 neurons x 785 inputs...")
    for n in range(16):
        if n % 4 == 0:
            dut._log.info(f"  Neuron {n}/16")
        for _ in range(785):
            await d.send_pair(1, 1, MODE_INFERENCE)
    await ClockCycles(dut.clk, 20)
    s = await d.status()
    assert (s >> 5) & 1, "Should be in Layer 2"
    dut._log.info("L2: 10 classes x 16 hidden...")
    for c in range(10):
        dut._log.info(f"  Class {c}/10")
        for h in range(16):
            assert await d.wait_ready(), f"Timeout at class {c}"
            w = 127 if c == 5 else 1
            await d.send_weight(w, MODE_INFERENCE)
            await ClockCycles(dut.clk, 12)
    await ClockCycles(dut.clk, 50)
    assert await d.wait_inf_done(100), "Timeout waiting for inf_done"
    result = await d.data()
    best = result & 0x0F
    dut._log.info(f"Predicted: {best}, Expected: 5")
    assert best == 5, f"Expected 5, got {best}"
    dut._log.info("PASS: test_inference")