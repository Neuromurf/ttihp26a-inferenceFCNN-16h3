"""
test.py - TinyTapeout GitHub Actions Compatible Testbench

Tests the tt_um_neuromurf_seq_mac_inf wrapper via tb.v.
No external files required - all data is hardcoded.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

# =============================================================================
# Constants
# =============================================================================
CLK_PERIOD_NS = 20  # 50 MHz

MODE_IDLE      = 0b00
MODE_MAC_ONLY  = 0b01
MODE_INFERENCE = 0b10

NUM_INPUTS  = 785
NUM_HIDDEN  = 16
NUM_CLASSES = 10

# Pin bit positions in uio_in
BIT_TOGGLE   = 0
BIT_TYPE     = 1
BIT_START    = 2
BIT_MODE0    = 3
BIT_NEXT     = 4
BIT_SOFT_RST = 5
BIT_MODE1    = 6
BIT_STATUS   = 7


# =============================================================================
# Helper Class
# =============================================================================
class DUT:
    """Simple driver for TinyTapeout wrapper."""
    
    def __init__(self, dut):
        self.dut = dut
        self.toggle = 0
        self.next_toggle = 1  # Start at 1, first toggle goes 1→0
    
    def _uio(self, toggle=None, dtype=0, start=0, mode=0, next_byte=None, soft_rst=0, status_sel=0):
        """Build uio_in value."""
        if toggle is None:
            toggle = self.toggle
        if next_byte is None:
            next_byte = self.next_toggle
        return (
            (toggle     << BIT_TOGGLE) |
            (dtype      << BIT_TYPE) |
            (start      << BIT_START) |
            ((mode & 1) << BIT_MODE0) |
            (next_byte  << BIT_NEXT) |
            (soft_rst   << BIT_SOFT_RST) |
            ((mode >> 1) << BIT_MODE1) |
            (status_sel << BIT_STATUS)
        )
    
    async def reset(self):
        """Reset the design."""
        self.dut.ena.value = 1
        self.dut.ui_in.value = 0
        self.dut.uio_in.value = 0
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, 5)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 5)
        self.toggle = 0
        self.next_toggle = 1  # Start at 1, first toggle goes 1→0
    
    async def status(self):
        """Read status byte."""
        self.dut.uio_in.value = self._uio(status_sel=1)
        await ClockCycles(self.dut.clk, 1)
        return int(self.dut.uo_out.value)
    
    async def data(self):
        """Read data byte."""
        self.dut.uio_in.value = self._uio(status_sel=0)
        await ClockCycles(self.dut.clk, 1)
        return int(self.dut.uo_out.value)
    
    async def wait_ready(self, timeout=100):
        """Wait for ready bit (status[2])."""
        for _ in range(timeout):
            s = await self.status()
            if s & 0x04:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False
    
    async def wait_done(self, timeout=100):
        """Wait for done bit (status[1])."""
        for _ in range(timeout):
            s = await self.status()
            if s & 0x02:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False
    
    async def wait_byte_valid(self, timeout=100):
        """Wait for byte_valid bit (status[3])."""
        for _ in range(timeout):
            s = await self.status()
            if s & 0x08:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False
    
    async def wait_inf_done(self, timeout=100):
        """Wait for inf_done bit (status[4])."""
        for _ in range(timeout):
            s = await self.status()
            if s & 0x10:
                return True
            await ClockCycles(self.dut.clk, 1)
        return False
    
    async def start_mode(self, mode):
        """Start operation in specified mode."""
        self.dut.uio_in.value = self._uio(start=1, mode=mode)
        await ClockCycles(self.dut.clk, 4)
        self.dut.uio_in.value = self._uio(start=0, mode=mode)
        await ClockCycles(self.dut.clk, 2)
    
    async def send_weight(self, weight, mode):
        """Send weight byte."""
        # Set data BEFORE toggle (RTL captures on edge)
        self.dut.ui_in.value = weight & 0xFF
        self.dut.uio_in.value = self._uio(dtype=0, mode=mode)
        await ClockCycles(self.dut.clk, 1)
        # Toggle
        self.toggle = 1 - self.toggle
        self.dut.uio_in.value = self._uio(dtype=0, mode=mode)
        await ClockCycles(self.dut.clk, 1)
    
    async def send_input(self, inp, mode):
        """Send input byte."""
        # Set data BEFORE toggle
        self.dut.ui_in.value = inp & 0xFF
        self.dut.uio_in.value = self._uio(dtype=1, mode=mode)
        await ClockCycles(self.dut.clk, 1)
        # Toggle
        self.toggle = 1 - self.toggle
        self.dut.uio_in.value = self._uio(dtype=1, mode=mode)
        await ClockCycles(self.dut.clk, 1)
    
    async def send_pair(self, weight, inp, mode):
        """Send weight-input pair."""
        assert await self.wait_ready(), "Timeout waiting for ready (weight)"
        await self.send_weight(weight, mode)
        assert await self.wait_ready(), "Timeout waiting for ready (input)"
        await self.send_input(inp, mode)
        await ClockCycles(self.dut.clk, 12)  # Wait for multiply
    
    async def next_byte(self, mode):
        """Shift to next result byte."""
        # Toggle and set immediately
        self.next_toggle = 1 - self.next_toggle
        self.dut.uio_in.value = self._uio(mode=mode, status_sel=0)
        await ClockCycles(self.dut.clk, 8)  # More margin for CDC + FSM + shift


# =============================================================================
# TEST: Reset
# =============================================================================
@cocotb.test()
async def test_reset(dut):
    """Test reset clears all flags."""
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    
    d = DUT(dut)
    await d.reset()
    
    s = await d.status()
    assert (s & 0x01) == 0, f"busy should be 0, got {s & 0x01}"
    assert (s & 0x40) == 0, f"err_flag should be 0, got {s & 0x40}"
    
    dut._log.info("PASS: test_reset")


# =============================================================================
# TEST: MAC Simple (9 × 1 × 1 = 9)
# =============================================================================
@cocotb.test()
async def test_mac_simple(dut):
    """Test MAC_ONLY: 9 × (1 × 1) = 9."""
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    
    d = DUT(dut)
    await d.reset()
    await d.start_mode(MODE_MAC_ONLY)
    
    # Send 9 pairs
    for _ in range(9):
        await d.send_pair(1, 1, MODE_MAC_ONLY)
    
    # Wait for result
    await d.wait_byte_valid(50)
    byte0 = await d.data()
    
    dut._log.info(f"MAC result: {byte0}, expected: 9")
    assert byte0 == 9, f"Expected 9, got {byte0}"
    dut._log.info("PASS: test_mac_simple")


# =============================================================================
# TEST: MAC Larger (9 × 10 × 20 = 1800)
# =============================================================================
@cocotb.test()
async def test_mac_larger(dut):
    """Test MAC_ONLY: 9 × (10 × 20) = 1800."""
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    
    d = DUT(dut)
    await d.reset()
    await d.start_mode(MODE_MAC_ONLY)
    
    # Send 9 pairs
    for _ in range(9):
        await d.send_pair(10, 20, MODE_MAC_ONLY)
    
    # Wait for byte_valid
    await d.wait_byte_valid(50)
    
    # Read byte0 (LSB)
    byte0 = await d.data()
    dut._log.info(f"byte0: {byte0}")
    
    # Shift and read byte1
    await d.next_byte(MODE_MAC_ONLY)
    byte1 = await d.data()
    dut._log.info(f"byte1: {byte1}")
    
    # Shift and read byte2
    await d.next_byte(MODE_MAC_ONLY)
    byte2 = await d.data()
    dut._log.info(f"byte2: {byte2}")
    
    result = byte0 | (byte1 << 8) | (byte2 << 16)
    dut._log.info(f"MAC result: {result} (bytes: {byte0}, {byte1}, {byte2}), expected: 1800")
    
    # 1800 = 0x708 → byte0=8, byte1=7, byte2=0
    assert result == 1800, f"Expected 1800, got {result}"
    dut._log.info("PASS: test_mac_larger")


# =============================================================================
# TEST: Inference
# =============================================================================
@cocotb.test()
async def test_inference(dut):
    """Test full inference with dummy weights.
    
    L1: w=1, x=1 → acc=785 → hidden=round(785/256)=3
    L2: class 5 gets w=127, others w=1
        class 5 score = 16×127×3 = 6096 (wins)
    """
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    
    d = DUT(dut)
    await d.reset()
    await d.start_mode(MODE_INFERENCE)
    
    dut._log.info("L1: 16 neurons × 785 inputs...")
    for n in range(NUM_HIDDEN):
        if n % 4 == 0:
            dut._log.info(f"  Neuron {n}/16")
        for _ in range(NUM_INPUTS):
            await d.send_pair(1, 1, MODE_INFERENCE)
    
    await ClockCycles(dut.clk, 20)
    
    # Verify L2
    s = await d.status()
    assert (s >> 5) & 1, "Should be in Layer 2"
    
    dut._log.info("L2: 10 classes × 16 hidden...")
    for c in range(NUM_CLASSES):
        dut._log.info(f"  Class {c}/10")
        for h in range(NUM_HIDDEN):
            assert await d.wait_ready(), f"Timeout at class {c}"
            w = 127 if c == 5 else 1
            await d.send_weight(w, MODE_INFERENCE)
            await ClockCycles(dut.clk, 12)
    
    await ClockCycles(dut.clk, 50)
    
    # Check result
    assert await d.wait_inf_done(100), "Timeout waiting for inf_done"
    result = await d.data()
    best = result & 0x0F
    
    dut._log.info(f"Predicted: {best}, Expected: 5")
    assert best == 5, f"Expected 5, got {best}"
    dut._log.info("PASS: test_inference")


# =============================================================================
# TEST: Soft Reset
# =============================================================================
@cocotb.test()
async def test_soft_reset(dut):
    """Test soft reset stops operation."""
    clock = Clock(dut.clk, CLK_PERIOD_NS, units="ns")
    cocotb.start_soon(clock.start())
    
    d = DUT(dut)
    await d.reset()
    await d.start_mode(MODE_INFERENCE)
    
    # Verify busy
    s = await d.status()
    assert s & 0x01, "Should be busy"
    
    # Soft reset
    d.dut.uio_in.value = d._uio(soft_rst=1)
    await ClockCycles(dut.clk, 5)
    d.dut.uio_in.value = d._uio(soft_rst=0)
    await ClockCycles(dut.clk, 5)
    
    # Verify idle
    s = await d.status()
    assert (s & 0x01) == 0, f"busy should be 0, got {s & 0x01}"
    dut._log.info("PASS: test_soft_reset")