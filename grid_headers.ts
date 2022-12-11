import { GImageSet, FIELDS, createElement, sort } from "./types.js"
import { GImageGrid } from "./grid.js"

class GridHeader {
    rowStart: number = 1
    rowEnd: number = 1
    values: Map<string, string | number>
    classes: Set<string>

    constructor(values: Map<string, string | number>, row: number) {
        this.values = values
        this.rowStart = this.rowEnd = row
        this.classes = new Set()
    }

    copy(): GridHeader {
        const res = new GridHeader(new Map(), 0)
        res.rowStart = this.rowStart
        res.rowEnd = this.rowEnd

        for (const key of this.values.keys()) {
            res.values.set(key, this.values.get(key)!)
        }

        return res
    }
}

class GridHeaders {
    grid: GImageGrid
    headers: GridHeader[]

    constructor(grid: GImageGrid) {
        this.grid = grid
        this.update()
    }

    update(): GridHeader[] {
        const headers = new Array<GridHeader>()   // return
        const lastValues = new Map<string, string | number>()

        var curHeader: GridHeader | null = null

        // build out rows with only the changing fields in them.
        for (const [isetIdx, isetKey] of this.grid.imageSetKeys.entries()) {
            const iset = this.grid.imageSets.get(isetKey)!

            // make no more than one new header per iset
            var madeNewHeader = false
            for (const [idx, field] of FIELDS.entries()) {
                if (field == "modelStr") {
                    continue;
                }

                const curValue = iset[field]!
                const lastValue = lastValues.get(field)

                if (field == "modelName") {
                    console.log(`madeNewHeader ${madeNewHeader}, curValue ${curValue}, lastValue ${lastValue}, curHeader.rowEnd ${curHeader?.rowEnd}`)
                }

                if (!madeNewHeader && (curHeader == null || curValue != lastValue)) {
                    const row = curHeader != null ? curHeader.rowEnd : 2
                    curHeader = new GridHeader(new Map(), row)
                    headers.push(curHeader)
                    madeNewHeader = true
                }

                if (lastValue != curValue) {
                    curHeader!.values.set(field, curValue)
                }
                lastValues.set(field, curValue)
            }
            curHeader!.rowEnd ++
        }

        // while values on a header are only what's different from previous, the classes are complete:
        // there is a class for each field including what's the same as the prior header.
        lastValues.clear()
        for (const header of headers) {
            for (const field of FIELDS) {
                const curValue = header.values.get(field) || lastValues.get(field)
                const valueIndex = this.grid.fieldValueIndex.get(field)?.get(curValue!)
                if (valueIndex != undefined) {
                    header!.classes.add(`${field}_${valueIndex}`)
                }
                else {
                    console.log(`can't find valueIndex for ${field}=${curValue}; all values = ${this.grid.fieldValueIndex.get(field)}`)
                }
                lastValues.set(field, curValue)
            }
        }

        this.headers = headers
        return this.headers
    }
}

export { GridHeaders }
