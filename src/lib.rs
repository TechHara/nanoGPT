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
            14, 10, 24, 11, 57, 60, 39, 8, 22, 12, 29, 35, 17, 6, 62, 16, 32, 28, 2, 43, 1, 15, 13,
            54, 48, 62, 39, 11, 45, 31, 31, 57, 56, 29, 60, 28, 48, 17, 5, 10, 53, 25, 33, 52, 63,
            11, 15, 58, 35, 36, 31, 39, 18, 8, 28, 10, 53, 16, 43, 52, 46, 28, 59, 23, 26, 37, 39,
            2, 23, 16, 33, 51, 0, 13, 51, 3, 21, 47, 32, 7, 12, 16, 4, 45, 24, 59, 13, 23, 63, 4,
            16, 18, 54, 12, 40, 19, 53, 2, 24, 22, 63, 60, 27, 11, 22, 8, 54, 16, 37, 29, 58, 14,
            14, 41, 45, 37, 40, 55, 6, 40, 56, 36, 52, 32, 0, 6, 35, 15, 2, 14, 20, 38, 57, 61, 19,
            24, 50, 9, 30, 7, 0, 55, 27, 30, 11, 35, 28, 20, 38, 4, 59, 59, 8, 31, 10, 60, 17, 28,
            36, 27, 32, 26, 9, 46, 51, 11, 47, 15, 61, 11, 41, 57, 32, 18, 17, 24, 23, 3, 6, 39,
            47, 18, 22, 26, 46, 16, 18, 17, 24, 23, 52, 39, 48, 6, 30, 19, 11, 21, 51, 50, 23, 11,
            19, 43, 25, 13, 39, 21, 26, 43, 7, 34, 10, 44, 58, 32, 42, 52, 58, 1, 54, 15, 49, 28,
            25, 8, 25, 31, 0, 61, 64, 55, 2, 50, 16, 25, 59,
        ];
        let c = compress(xs.clone())?;
        let d = decompress(c.clone())?;
        println! {"{:?}", c}
        assert_eq!(xs, d);
        Ok(())
    }
}
