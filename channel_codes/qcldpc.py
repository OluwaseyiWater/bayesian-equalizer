import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np, urllib.request, json
import importlib.util

LLR_MAX = 32.0

# Helper: find lifting size
def _find_lifting_size(k, k_b):
    Z_values = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52,
        56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
        160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384
    ]
    for Z in Z_values:
        if k_b * Z >= k:
            return Z
    raise ValueError(f"Could not find suitable Z for k={k}")

# 5G NR Base Graph 1
def _get_5g_nr_bg1_base_matrix():
    try:
        module_path = os.path.join(os.path.dirname(__file__), "bg1_matrix.py")

        if not os.path.exists(module_path):
            raise FileNotFoundError(
                f"[ldpc] Missing bg1_matrix.py at {module_path}. "
                "Please ensure the file exists with BG1_BASE defined."
            )

        spec = importlib.util.spec_from_file_location("bg1_matrix", module_path)
        bg1_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bg1_module)

        base_graph = np.array(bg1_module.BG1_BASE, dtype=int)

        if base_graph.shape != (46, 68):
            raise ValueError(f"[ldpc] Invalid BG1 matrix shape {base_graph.shape}, expected (46, 68).")

        print(f"[ldpc] Loaded local 5G NR BG1 base matrix ({base_graph.shape}).")
        k_b = 22
        return base_graph, k_b

    except Exception as e:
        raise RuntimeError(f"[ldpc] Failed to load BG1 matrix: {e}")


# Generator matrix
def _get_generator_matrix_systematic(H, k, n):
    raise NotImplementedError(
        "Generator matrix construction disabled: current BG1 matrix is not verified. "
        "Decoding works; encoding is unavailable until the base graph is corrected."
    )



# QCLDPCCode class
class QCLDPCCode:
    def __init__(self, k, **kwargs):
        print("[ldpc] Using 5G NR QC-LDPC (BG1) structure.")
        self.H_base, self.k_b = _get_5g_nr_bg1_base_matrix()
        self.Z = _find_lifting_size(k, self.k_b)

        self.k_actual = self.k_b * self.Z
        self.n_actual = self.H_base.shape[1] * self.Z

        if k != self.k_actual:
            print(f"[ldpc] Warning: message length k={k} not directly supported.")
            print(f"[ldpc] Using k={self.k_actual} (Z={self.Z}) instead.")

        self.k, self.n = self.k_actual, self.n_actual
        self.rate = self.k / self.n

        # build H
        rows_base, cols_base = self.H_base.shape
        self.H = np.zeros((rows_base * self.Z, cols_base * self.Z), dtype=np.uint8)
        for i in range(rows_base):
            for j in range(cols_base):
                shift = self.H_base[i, j]
                if shift != -1:
                    # In 3GPP TS 38.212 the circulant permutation Q(P) is defined by
                    # circularly shifting the identity matrix to the **right** P times.
                    # np.roll with a negative shift rolls to the right; np.roll(I, -P,
                    # axis=1) yields a circulant such that Q(P) @ v = circ_shift_right(v, P).
                    I = np.eye(self.Z, dtype=np.uint8)
                    self.H[i*self.Z:(i+1)*self.Z, j*self.Z:(j+1)*self.Z] = np.roll(I, -shift % self.Z, axis=1)

        print(f"[ldpc] H shape: {self.H.shape}, expected ({rows_base*self.Z}, {cols_base*self.Z})")
        # Determine the lifting-size set index (iLS) and choose the parity structure variant.
        # 3GPP TS 38.212 defines eight sets of allowable Z values for BG1 (set index 0..7).
        # BG1_B1 is used for iLS ∈ {0,1,2,3,4,5,7}, while BG1_B2 is used for iLS = 6【54131343894641†L104-L107】.
        BG1_Z_SETS = {
            0: [2, 4, 8, 16, 32, 64, 128, 256],
            1: [3, 6, 12, 24, 48, 96, 192, 384],
            2: [5, 10, 20, 40, 80, 160, 320],
            3: [7, 14, 28, 56, 112, 224],
            4: [9, 18, 36, 72, 144, 288],
            5: [11, 22, 44, 88, 176, 352],
            6: [13, 26, 52, 104, 208],
            7: [15, 30, 60, 120, 240],
        }
        set_index = None
        for idx, values in BG1_Z_SETS.items():
            if self.Z in values:
                set_index = idx
                break
        # Default to 0 if not found.
        if set_index is None:
            set_index = 0
        # Choose parity structure: 'B2' for iLS=6, else 'B1'
        self._b_variant = 'B2' if set_index == 6 else 'B1'

        self.G = None
        self.info_cols = np.arange(self.k)

        self.m, self.n = self.H.shape

        # Precompute lists of check indices per variable and variable indices per check for decoding.
        self.checks_of_var = [np.where(self.H[:, j])[0].astype(np.int32) for j in range(self.n)]
        self.vars_of_check = [np.where(self.H[i, :])[0].astype(np.int32) for i in range(self.m)]

    def _circ_shift_right(self, vec, shift):
        """
        Perform a circular right shift by `shift` positions on a 1‑D uint8 array.
        A shift of 0 returns a copy.  If shift is negative, left shift.
        """
        # ensure shift is within [0, Z) where Z is len(vec)
        Z = vec.size
        if shift == 0 or shift % Z == 0:
            return vec.copy()
        return np.roll(vec, shift % Z)

    def _encode_systematic(self, msg_bits):
        """
        Internal helper implementing 5G NR QC‑LDPC systematic encoding for BG1.

        This constructs the full codeword [s | p] given the message bits `msg_bits`
        such that H * codeword^T = 0 (mod 2).  The encoding follows the
        high‑level algorithm described in 3GPP TS 38.212 and [Lyons 2019],
        assuming the BG1_B1 parity structure.  The message bits are divided
        into kb blocks of length Z and padded with zeros if necessary.  Four
        parity blocks (pb1..pb4) are computed from the first four base rows
        using the relations:

            λ_i = sum_{j=1..kb} a[i,j] * s_j  (over GF(2))
            pb1 = λ1 ⊕ λ2 ⊕ λ3 ⊕ λ4
            pb2 = λ1 ⊕ shift_right(pb1, 1)
            pb4 = λ4 ⊕ shift_right(pb1, 1)
            pb3 = λ3 ⊕ pb4

        where shift_right(x, k) denotes a cyclic right shift by k.  The
        remaining parity blocks pc_i for i ∈ {1..mb−4} are computed by

            pc_i = sum_{j=1..kb} c1[i,j] * s_j  ⊕  sum_{b=1..4} c2[i,b] * pb_b

        with all operations in GF(2).  Finally the codeword is assembled as
        [s | pb | pc].  The first 2×Z systematic bits are not punctured here.
        """
        # Determine lifting size and base dimensions
        Z = self.Z
        H_base = self.H_base
        mb, nb = H_base.shape
        kb = self.k_b

        # Prepare the message vector of length kb*Z (pad with zeros if needed)
        msg = np.zeros(kb * Z, dtype=np.uint8)
        mlen = min(msg_bits.size, kb * Z)
        msg[:mlen] = msg_bits[:mlen]

        # Partition message into kb blocks of length Z
        s_blocks = [msg[i*Z:(i+1)*Z] for i in range(kb)]

        # Compute lambda_i for i=0..3 (rows 0..3)
        lambdas = []
        for row in range(4):
            vec = np.zeros(Z, dtype=np.uint8)
            for j in range(kb):
                shift = int(H_base[row, j])
                if shift >= 0:
                    # Add Q(shift) * s_j (right shift by shift)
                    vec ^= self._circ_shift_right(s_blocks[j], shift)
            lambdas.append(vec)

        # Compute parity blocks pb1..pb4 depending on the BG1 parity variant.
        # For BG1_B1 (most lifting sizes), the relations are【54131343894641†L145-L150】:
        #   pb1 = λ1 ⊕ λ2 ⊕ λ3 ⊕ λ4
        #   pb2 = λ1 ⊕ p(1) pb1
        #   pb4 = λ4 ⊕ p(1) pb1
        #   pb3 = λ3 ⊕ pb4
        # where p(α) denotes a cyclic right shift by α positions.  For BG1_B2 (iLS=6),
        # we use【54131343894641†L145-L150】:
        #   p(105 mod Z) pb1 = λ1 ⊕ λ2 ⊕ λ3 ⊕ λ4   → pb1 = p(−105 mod Z)(λ1 ⊕ λ2 ⊕ λ3 ⊕ λ4)
        #   pb2 = λ1 ⊕ pb1
        #   pb4 = λ4 ⊕ pb1
        #   pb3 = λ3 ⊕ pb4
        lambda_sum = lambdas[0].copy()
        lambda_sum ^= lambdas[1]
        lambda_sum ^= lambdas[2]
        lambda_sum ^= lambdas[3]
        if self._b_variant == 'B1':
            pb1 = lambda_sum
            pb2 = lambdas[0] ^ self._circ_shift_right(pb1, 1)
            pb4 = lambdas[3] ^ self._circ_shift_right(pb1, 1)
            pb3 = lambdas[2] ^ pb4
        else:
            # BG1_B2: pb1 is the inverse shift of 105 positions of the sum of lambdas.
            shift_amt = 105 % Z
            # p(shift_amt) pb1 = lambda_sum → pb1 = shift left by shift_amt
            pb1 = np.roll(lambda_sum, -shift_amt)
            # pb2, pb4, pb3 without additional shift
            pb2 = lambdas[0] ^ pb1
            pb4 = lambdas[3] ^ pb1
            pb3 = lambdas[2] ^ pb4
        pb_blocks = [pb1, pb2, pb3, pb4]

        # Compute pc_i for rows 4..mb-1
        pc_blocks = []
        for row in range(4, mb):
            vec = np.zeros(Z, dtype=np.uint8)
            # C1 part: contributions from s_blocks
            for j in range(kb):
                shift = int(H_base[row, j])
                if shift >= 0:
                    vec ^= self._circ_shift_right(s_blocks[j], shift)
            # C2 part: contributions from pb blocks
            for b_idx in range(4):
                shift = int(H_base[row, kb + b_idx])
                if shift >= 0:
                    vec ^= self._circ_shift_right(pb_blocks[b_idx], shift)
            pc_blocks.append(vec)

        # Assemble codeword blocks [s | pb | pc]
        blocks = s_blocks + pb_blocks + pc_blocks
        codeword = np.concatenate(blocks).astype(np.uint8)
        return codeword

    def encode(self, msg_bits):
        """
        Encode a message bit sequence using the QC‑LDPC parity‑check matrix.

        This implementation pads or truncates the input to match the code
        dimension `self.k` and then calls the systematic QC‑LDPC encoder.
        """
        msg_bits = np.asarray(msg_bits, dtype=np.uint8) & 1
        if msg_bits.size > self.k:
            raise ValueError(
                f"Message length {msg_bits.size} exceeds code dimension {self.k}")
        # If the message is shorter than k, pad zeros to length k
        if msg_bits.size < self.k:
            padded = np.zeros(self.k, dtype=np.uint8)
            padded[: msg_bits.size] = msg_bits
            msg_bits = padded
        return self._encode_systematic(msg_bits)

    def decode_extrinsic(self, L_a, iters=40, mode="nms", alpha=0.85, damping=0.0, early_stop=True):
        m, n = self.m, self.n
        L_a = np.clip(np.asarray(L_a, dtype=float), -LLR_MAX, LLR_MAX)
        a = float(alpha)
        dmp = float(damping)

        L_vc = [{int(ci): float(L_a[j]) for ci in self.checks_of_var[j]} for j in range(n)]
        L_cv = [{int(vj): 0.0 for vj in self.vars_of_check[i]} for i in range(m)]

        for it in range(int(iters)):
            # ----- Check node update (C -> V) -----
            for i in range(m):
                vs = self.vars_of_check[i]
                if vs.size == 0: continue
                
                msgs = np.array([L_vc[v][i] for v in vs], dtype=float)
                msgs = np.clip(msgs, -LLR_MAX, LLR_MAX)

                sgn = np.sign(msgs); sgn[sgn == 0] = 1.0
                prod_sgn = np.prod(sgn)
                aabs = np.abs(msgs)

                idx_min = int(np.argmin(aabs))
                min1 = aabs[idx_min]
                
                tmp = aabs.copy(); tmp[idx_min] = np.inf
                min2 = float(np.min(tmp))

                for t, v in enumerate(vs):
                    mag = min2 if t == idx_min else min1
                    msg_new = a * prod_sgn * sgn[t] * mag
                    msg_new = float(np.clip(msg_new, -LLR_MAX, LLR_MAX))
                    if dmp > 0.0:
                        L_cv[i][int(v)] = (1 - dmp) * msg_new + dmp * L_cv[i][int(v)]
                    else:
                        L_cv[i][int(v)] = msg_new

            # ----- Variable node update (V -> C) -----
            for j in range(n):
                cs = self.checks_of_var[j]
                if cs.size == 0: continue
                
                incoming = [np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX) for ci in cs]
                total = float(np.clip(L_a[j] + sum(incoming), -LLR_MAX, LLR_MAX))
                
                for ci in cs:
                    val = float(np.clip(total - np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX), -LLR_MAX, LLR_MAX))
                    if dmp > 0.0:
                        L_vc[j][int(ci)] = (1 - dmp) * val + dmp * L_vc[j][int(ci)]
                    else:
                        L_vc[j][int(ci)] = val

            # ----- A-posteriori LLRs and early stopping -----
            L_post = np.zeros(n, dtype=float)
            for j in range(n):
                L_post[j] = float(np.clip(L_a[j] + sum(np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX)
                                                       for ci in self.checks_of_var[j]), -LLR_MAX, LLR_MAX))
            if early_stop:
                hard = (L_post < 0).astype(np.uint8)
                synd = (self.H @ hard) & 1
                if not np.any(synd):
                    break
        
        # After completing BP iterations, compute a‑posteriori and extrinsic LLRs.
        # Following the 3GPP reference implementation (and the baseline LDPC),
        # the extrinsic bit LLRs are defined as the difference between the
        # a‑posteriori LLRs and the a‑priori LLRs.  This provides the decoder’s
        # new information about each code bit.  See LDPC.decode_extrinsic() for
        # the reference behaviour.  Summing only check‑to‑variable messages,
        # without subtracting L_a, would incorrectly bias the extrinsic values
        # and degrade turbo equalization performance.
        L_post = np.clip(L_post, -LLR_MAX, LLR_MAX)
        # Extrinsic LLRs for each code bit
        L_ext = np.clip(L_post - L_a, -LLR_MAX, LLR_MAX)
        # A‑posteriori LLRs for the message bits (information positions)
        L_post_msg = L_post[self.info_cols]
        return L_post_msg, L_ext