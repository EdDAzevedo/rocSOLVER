
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
 // -------------------------------------------------
 // inline lambda function to perform search in array
 // -------------------------------------------------
auto rf_search = [](Iint const len, Iint const* const arr, Iint const key) -> Iint {
         // ---------------------------------------
         // search array  arr[0], ..., arr[ len-1] 
         // for matching value "key"
	 // 
         // return the index value of matching position
         // ---------------------------------------
    Iint constexpr small_len = 8;
    Iint ipos = len;
    if((len <= 0) || (arr == nullptr))
    {
        return (ipos = len);
    };

          // -----------------
          // use binary search
          // -----------------
    Iint lo = 0;
    Iint hi = len;

    for(int i = 0; i < 32; i++)
    {
        Iint const len_remain = hi - lo;
        if(len_remain <= small_len)
        {
                 // ------------------------
                 // use simple linear search
                 // ------------------------
            for(int k = 0; k < len; k++)
            {
                bool const is_found = (arr[k] == key);
                if(is_found)
                {
                    return (ipos = k);
                };
            };
        }
        else
        {
            Iint const mid = (lo + hi) / 2;
            bool const is_found = (arr[mid] == key);
            if(is_found)
            {
                return (ipos = mid);
            };

            if(arr[mid] < key)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            };
        };
    };
    return (ipos);
};
