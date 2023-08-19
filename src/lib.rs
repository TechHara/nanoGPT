use pyo3::{exceptions::PyTypeError, prelude::*};

type Number = i64;
type Pos = u16;
const MASK: usize = NUM_HASH - 1;
const NUM_HASH: usize = 1 << 9; // 9 bit or 512

// x from 0 to 64 inclusive, i.e., 65 tokens
const OFFSET_8: Number = 65;
const OFFSET_1: Number = OFFSET_8 + 8; // 73
const LENGTH: Number = OFFSET_1 + 8; // 81
const NUM_TOKENS: Number = LENGTH + 16; // 98
const MAX_LENGTH: usize = 16;

#[pyfunction]
fn compress(xs: Vec<Number>) -> PyResult<Vec<Number>> {
    if xs.len() < 3 {
        return Ok(xs);
    }

    let mut result = Vec::with_capacity(xs.len());
    let mut chain = HashChain::new(&xs[..], 64);
    let mut src_pos = 0;
    let mut iter = xs.iter().skip(2);
    while let Some(x) = iter.next() {
        if let Some(match_pos) = chain.insert(*x) {
            let length_pos = chain.get_max_length(match_pos, src_pos as Pos);
            match length_pos.0 {
                0..=3 => {
                    result.push(xs[src_pos]);
                    src_pos += 1;
                }
                length => {
                    let offset = (src_pos - length_pos.1) as Number;
                    result.push(offset / 8 + OFFSET_8);
                    result.push(offset % 8 + OFFSET_1);
                    result.push(length as Number + LENGTH);
                    for _ in 0..length {
                        iter.next().map(|x| chain.insert(*x));
                        src_pos += 1;
                    }
                }
            }
        } else {
            result.push(xs[src_pos]);
            src_pos += 1;
        }
    }
    for x in &xs[src_pos..] {
        result.push(*x);
    }
    Ok(result)
}

#[pyfunction]
fn decompress(xs: Vec<Number>) -> PyResult<Vec<Number>> {
    let mut result = Vec::new();
    let mut iter = xs.iter();
    while let Some(x) = iter.next() {
        let x = *x;
        if x < OFFSET_8 {
            // literal
            result.push(x);
        } else if x < OFFSET_1 {
            let mut offset = (x - OFFSET_8) << 3;
            match (iter.next(), iter.next()) {
                (Some(y), Some(z))
                    if *y >= OFFSET_1 && *y < LENGTH && *z >= LENGTH && *z <= NUM_TOKENS =>
                {
                    offset += *y - OFFSET_1;
                    let length = *z - LENGTH;
                    for _ in 0..length {
                        result.push(result[result.len() - offset as usize]);
                    }
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Decompressor Error: Invalid error {}...",
                        x
                    )))
                }
            }
        }
    }

    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn compressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    Ok(())
}

struct HashChain<'a> {
    h: usize,
    head: Vec<Pos>,
    tail: Vec<Pos>,
    cur_pos: Pos,
    max_offset: usize,
    xs: &'a [Number],
}

impl HashChain<'_> {
    fn new(xs: &[Number], max_offset: usize) -> HashChain {
        let h = (((xs[0] as usize) << 3) ^ (xs[1] as usize)) & MASK;
        HashChain {
            h,
            head: vec![0 as Pos; NUM_HASH],
            tail: vec![0 as Pos; xs.len()],
            cur_pos: 0,
            max_offset,
            xs,
        }
    }

    // update hash
    // update head & tail
    // returns pos pointed by head
    fn insert(&mut self, x: Number) -> Option<Pos> {
        self.update_hash(x);
        let prev_pos = self.head[self.h];
        self.head[self.h] = self.cur_pos;
        self.tail[self.cur_pos as usize] = prev_pos;
        self.cur_pos += 1;
        if prev_pos == 0 {
            None
        } else {
            Some(prev_pos)
        }
    }

    fn update_hash(&mut self, x: Number) {
        self.h = ((self.h << 3) ^ (x as usize)) & MASK;
    }

    fn next_pos(&self, pos: Pos) -> Option<Pos> {
        match &self.tail[pos as usize] {
            0 => None,
            pos => Some(*pos),
        }
    }

    // returns (max length, pos)
    fn get_max_length(&self, mut pos: Pos, target_pos: Pos) -> (usize, usize) {
        let mut result = (0, 0);
        let target = &self.xs[target_pos as usize..];
        loop {
            if target_pos as usize - pos as usize > self.max_offset {
                break;
            }
            let length = self.xs[pos as usize..]
                .iter()
                .zip(target.iter())
                .map_while(|(x, y)| match x.eq(y) {
                    true => Some(true),
                    false => None,
                })
                .count();
            result = if length > result.0 {
                (std::cmp::min(length, MAX_LENGTH), pos as usize)
            } else {
                result
            };

            pos = if let Some(match_pos) = self.next_pos(pos) {
                match_pos
            } else {
                break;
            }
        }

        result
    }
}

#[cfg(test)]
mod test {
    type Result<R> = std::result::Result<R, Box<dyn std::error::Error>>;
    use crate::OFFSET_8;

    use super::compress;
    use super::decompress;
    use rand::prelude::*;

    #[test]
    fn test() -> Result<()> {
        let x = vec![0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6];
        let compressed = compress(x.clone())?;
        let decompressed = decompress(compressed)?;
        assert_eq!(x, decompressed);
        Ok(())
    }

    #[test]
    fn test_random() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        let mut rng = thread_rng();
        let distr = rand::distributions::Uniform::new(0, OFFSET_8);
        let length_distr = rand::distributions::Uniform::new(0, 256);
        for _ in 0..100000 {
            let n = rng.sample(length_distr);
            let mut xs = Vec::new();
            for _ in 0..n {
                xs.push(rng.sample(distr));
            }

            let same = decompress(compress(xs.clone())?)? == xs;
            if !same {
                println!("{:?}", xs);
            }
        }

        Ok(())
    }

    #[test]
    fn test_case() -> Result<()> {
        let xs = vec![
            51, 51, 30, 40, 24, 38, 39, 49, 12, 60, 35, 40, 10, 48, 64, 60, 5, 10, 18, 61, 35, 10,
            20, 21, 10, 50, 31, 48, 26, 32, 42, 6, 11, 43, 5, 41, 57, 0, 48, 32, 14, 36, 38, 31,
            47, 7, 19, 59, 25, 40, 58, 27, 36, 5, 60, 42, 1, 21, 21, 3, 23, 35, 51, 8, 17, 21, 11,
            45, 34, 56, 59, 50, 27, 4, 18, 27, 15, 58, 16, 16, 14, 59, 22, 56, 38, 17, 40, 38, 17,
            40, 38, 43, 41, 52, 22, 27, 10, 61, 19, 56, 5, 0, 28, 1, 31, 4, 51, 38, 27, 61, 56, 63,
            19, 20, 60, 30, 57, 52, 23, 26, 11, 63, 35, 49, 25, 6, 13, 62, 48, 11, 36, 9, 38, 39,
            32, 56, 57, 28, 51, 27, 27, 9, 60, 11, 12, 2, 62, 46, 16, 63, 26, 28, 20, 41, 64, 48,
            0, 21, 24, 24, 4, 9, 49, 39, 63, 20, 18, 38, 47, 13, 62, 28, 11, 33, 3, 61, 23, 34, 18,
            22, 11, 56, 0, 1, 5, 11, 35, 57, 45, 47, 31, 50, 60, 43, 31, 39, 54, 35, 62, 50, 25, 3,
            17, 19, 62, 60, 3, 19, 2, 56, 25, 34, 3, 24, 36, 34, 29, 26, 39, 45, 57, 30, 27, 47, 8,
            24, 57, 3, 64, 20, 31, 20, 1, 30, 25, 43, 46, 0, 56, 9, 42, 51, 0, 61, 51, 50, 25, 3,
            10,
        ];
        let c = compress(xs.clone())?;
        let d = decompress(c.clone())?;
        println! {"{:?}", c}
        assert_eq!(xs, d);
        Ok(())
    }
}
