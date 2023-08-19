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
    } else if xs.len() > Pos::MAX as usize {
        return Err(PyTypeError::new_err(format!(
            "Input size too large: {}",
            xs.len()
        )));
    }

    let mut result = Vec::with_capacity(xs.len());
    let mut chain = HashChain::new(&xs[..], 64);
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
                    result.push(offset / 8 + OFFSET_8);
                    result.push(offset % 8 + OFFSET_1);
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
            0, 13, 57, 1, 51, 63, 1, 46, 39, 60, 43, 1, 52, 53, 58, 46, 43, 56, 1, 52, 53, 1, 50,
            53, 56, 42, 47, 43, 42, 6, 71, 74, 85, 53, 56, 1, 52, 53, 71, 79, 86, 58, 56, 59, 58,
            59, 44, 58, 1, 40, 53, 59, 56, 57, 8, 0, 0, 31, 43, 41, 53, 52, 57, 10, 0, 21, 1, 61,
            46, 39, 58, 1, 57, 47, 56, 6, 1, 61, 46, 43, 52, 1, 43, 60, 43, 56, 1, 58, 53, 53, 6,
            1, 46, 39, 58, 46, 1, 46, 47, 57, 1, 50, 47, 43, 44, 59, 50, 1, 58, 53, 1, 40, 43, 50,
            50, 8, 0, 37, 43, 58, 1, 40, 43, 1, 53, 41, 46, 43, 43, 56, 57, 1, 46, 43, 39, 56, 58,
            1, 44, 56, 53, 51, 1, 46, 66, 78, 85, 6, 0, 35, 47, 42, 53, 56, 43, 52, 1, 39, 52, 42,
            1, 40, 43, 1, 50, 47, 44, 43, 12, 0, 0, 32, 46, 43, 1, 54, 56, 39, 63, 10, 0, 20, 43,
            56, 43, 50, 47, 39, 52, 41, 43, 57, 6, 1, 58, 56, 53, 52, 45, 1, 46, 43, 39, 56, 1, 51,
            43, 1, 47, 52, 1, 53, 59, 56, 58, 1, 47, 52, 1, 53, 52, 43, 8, 0, 0, 15, 24, 13, 30,
            17, 26, 15, 17, 10, 0, 20, 39, 42, 1, 58, 46, 47, 57, 1, 63, 53, 59, 6, 1, 53, 52, 1,
            21, 1, 54, 39, 41, 43, 6, 1, 39, 57, 1, 63, 53, 59, 1, 41, 39, 52, 52, 53, 58, 1, 58,
            46, 53, 59, 45, 46, 0, 32, 46, 43, 1, 41, 39, 47, 50, 50, 69, 75, 86, 54, 53, 61, 43,
            56, 8, 1, 13, 52, 42, 71, 74, 85, 41, 53, 51, 51, 43, 52, 58, 0, 13, 57, 1, 58, 56, 59,
            43, 1, 63, 43, 39, 11, 1, 39, 52, 42, 7, 7, 61, 47, 58, 46, 43, 56, 1, 61, 53, 56, 58,
            46, 53, 56, 57, 6, 0, 35, 46, 47, 41, 46, 1, 46, 47, 51, 1, 57, 53, 53, 52, 1, 39, 1,
            41, 53, 52, 58, 50, 43, 1, 58, 46, 39, 52, 10, 0, 31, 58, 39, 52, 58, 6, 1, 61, 47, 50,
            50, 1, 52, 53, 58, 43, 69, 74, 86, 57, 46, 39, 50, 50, 1, 52, 53, 58, 1, 57, 59, 57,
            43, 1, 45, 56, 39, 41, 49, 6, 1, 51, 39, 49, 43, 1, 48, 53, 47, 57, 43, 8, 0, 0, 30,
            21, 27, 24, 13, 26, 33, 31, 10, 0, 14, 59, 58, 1, 5, 58, 47, 57, 1, 40, 39, 52, 1, 58,
            39, 49, 43, 1, 58, 56, 39, 47, 50, 1, 57, 46, 39, 50, 50, 1, 39, 50, 50, 1, 58, 46, 43,
            1, 54, 50, 43, 43, 55, 59, 43, 1, 47, 58, 8, 0, 27, 1, 50, 39, 63, 1, 51, 39,
        ];
        let d = decompress(xs.clone())?;
        Ok(())
    }
}
