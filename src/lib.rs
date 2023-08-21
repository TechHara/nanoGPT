use pyo3::{exceptions::PyTypeError, prelude::*};

type Number = i64;
type Pos = u16;
const MASK: usize = NUM_HASH - 1;
const NUM_HASH: usize = 1 << 9; // 9 bit or 512

// x from 0 to 64 inclusive, i.e., 65 tokens
const MAX_OFFSET: usize = 1024;
const BASE: Number = 32;
const OFFSET_32: Number = 65;
const OFFSET_1: Number = OFFSET_32 + BASE; // 81
const LENGTH: Number = OFFSET_1 + BASE; // 97
const NUM_TOKENS: Number = LENGTH + MAX_LENGTH as Number; // 113
const MAX_LENGTH: usize = 16;

#[pyfunction]
fn compress(xs: Vec<Number>) -> PyResult<Vec<Number>> {
    if xs.len() < 3 {
        return Ok(xs);
    } else if xs.len() > Pos::MAX as usize {
        return Err(PyTypeError::new_err(format!(
            "Input size too large: {}",
            xs.len()
        )));
    }

    let mut result = Vec::with_capacity(xs.len());
    let mut chain = HashChain::new(&xs[..], MAX_OFFSET);
    let mut src_pos = 0;
    let mut iter = xs.iter().skip(2);
    while let Some(x) = iter.next() {
        if chain.cur_pos != src_pos {
            println!("positions out of sync");
        }
        if let Some(match_pos) = chain.insert(*x) {
            let length_pos = chain.get_max_length(match_pos, src_pos as Pos);
            match length_pos.0 {
                0..=3 => {
                    result.push(xs[src_pos as usize]);
                    src_pos += 1;
                }
                length => {
                    let offset = (src_pos as usize - length_pos.1) as Number;
                    result.push(offset / BASE + OFFSET_32);
                    result.push(offset % BASE + OFFSET_1);
                    result.push(length as Number + LENGTH);
                    for _ in 0..length - 1 {
                        iter.next().map(|x| chain.insert(*x));
                        src_pos += 1;
                    }
                    src_pos += 1;
                }
            }
        } else {
            result.push(xs[src_pos as usize]);
            src_pos += 1;
        }
    }
    for x in &xs[src_pos as usize..] {
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
        if x < OFFSET_32 {
            // literal
            result.push(x);
        } else if x < OFFSET_1 {
            let mut offset = (x - OFFSET_32) * BASE;
            match (iter.next(), iter.next()) {
                (Some(y), Some(z))
                    if *y >= OFFSET_1 && *y < LENGTH && *z >= LENGTH && *z <= NUM_TOKENS =>
                {
                    offset += *y - OFFSET_1;
                    let length = *z - LENGTH;
                    if offset > result.len() as Number {
                        for _ in 0..length {
                            result.push(36); // X
                        }
                    } else {
                        for _ in 0..length {
                            result.push(result[result.len() - offset as usize]);
                        }
                    }
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Decompressor Error: Invalid error {:?}",
                        xs
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
    use crate::OFFSET_32;

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
        let distr = rand::distributions::Uniform::new(0, OFFSET_32);
        let length_distr = rand::distributions::Uniform::new_inclusive(0, 1024);
        for _ in 0..10000 {
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
            15, 20, 14, 58, 48, 31, 13, 50, 62, 54, 54, 27, 24, 44, 28, 19, 48, 22, 5, 9, 44, 16,
            36, 57, 28, 17, 58, 23, 4, 47, 26, 0, 64, 54, 42, 32, 21, 30, 18, 31, 10, 29, 52, 52,
            10, 43, 27, 15, 21, 9, 14, 20, 6, 8, 30, 58, 24, 42, 37, 4, 63, 62, 63, 38, 2, 10, 14,
            33, 50, 57, 56, 37, 31, 31, 2, 21, 16, 52, 38, 23, 25, 57, 12, 26, 11, 6, 59, 14, 41,
            62, 61, 39, 2, 63, 8, 58, 57, 36, 61, 20, 46, 32, 47, 7, 54, 19, 40, 16, 48, 55, 19,
            62, 29, 2, 45, 42, 1, 23, 14, 57, 15, 41, 1, 48, 60, 54, 47, 62, 30, 42, 12, 30, 36,
            11, 39, 2, 46, 5, 39, 48, 16, 30, 46, 1, 15, 44, 20, 48, 14, 58, 48, 31, 6, 40, 39, 1,
            34, 45, 41, 24, 38, 24, 46, 40, 56, 56, 14, 43, 60, 24, 52, 35, 51, 7, 32, 0, 25, 40,
            62, 12, 38, 38, 47, 40, 52, 63, 63, 5, 26, 22, 30, 53, 27, 7, 25, 46, 11, 26, 40, 19,
            45, 52, 31, 31, 36, 14, 24, 60, 5, 49, 7, 30, 14, 19, 27, 58, 37, 9, 52, 11, 24, 64,
            15, 44, 30, 15, 38, 56, 4, 16, 26, 6, 46, 48, 50, 43, 16, 6, 49, 58, 19, 22, 6, 63, 20,
            30, 0, 35, 59,
        ];
        let c = compress(xs.clone())?;
        let d = decompress(c.clone())?;
        assert_eq!(xs, d);
        Ok(())
    }
}
