(module
  (memory (export "memory") 1)
  (func (export "matvec") (param $q i32) (param $s i32) (param $x i32) (param $out i32) (param $r i32) (param $c i32)
    (local $i i32) (local $j i32) (local $c8 i32) (local $xp i32) (local $a v128) (local $b v128) (local $w v128) (local $sum f32)
    (local.set $c8 (i32.and (local.get $c) (i32.const -8)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $i) (local.get $r)))
        (local.set $a (v128.const f32x4 0 0 0 0))
        (local.set $b (v128.const f32x4 0 0 0 0))
        (local.set $j (i32.const 0))
        (block $eights
          (loop $cols
            (br_if $eights (i32.ge_u (local.get $j) (local.get $c8)))
            (local.set $w (v128.load8x8_s (i32.add (local.get $q) (local.get $j))))
            (local.set $xp (i32.add (local.get $x) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $a (f32x4.add (local.get $a) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $w)))
                                                                (v128.load (local.get $xp)))))
            (local.set $b (f32x4.add (local.get $b) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $w)))
                                                                (v128.load offset=16 (local.get $xp)))))
            (local.set $j (i32.add (local.get $j) (i32.const 8)))
            (br $cols)))
        (local.set $a (f32x4.add (local.get $a) (local.get $b)))
        (local.set $sum (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a)) (f32x4.extract_lane 1 (local.get $a)))
                                 (f32.add (f32x4.extract_lane 2 (local.get $a)) (f32x4.extract_lane 3 (local.get $a)))))
        (block $rest
          (loop $ones
            (br_if $rest (i32.ge_u (local.get $j) (local.get $c)))
            (local.set $sum (f32.add (local.get $sum) (f32.mul (f32.convert_i32_s (i32.load8_s (i32.add (local.get $q) (local.get $j))))
                                                                (f32.load (i32.add (local.get $x) (i32.shl (local.get $j) (i32.const 2)))))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $ones)))
        (f32.store (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))
                   (f32.mul (local.get $sum) (f32.load (i32.add (local.get $s) (i32.shl (local.get $i) (i32.const 2))))))
        (local.set $q (i32.add (local.get $q) (local.get $c)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $rows)))))
