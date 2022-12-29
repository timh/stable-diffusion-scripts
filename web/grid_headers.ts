import { GImageSet, FIELDS } from "./types.js"
import { createElement, sort } from "./util.js"
import { GImageGrid, STORE_HIDDEN } from "./grid.js"

class GridHeader {
    rowStart: number = 1
    rowEnd: number = 1
    values: Map<string, string | number>

    constructor(values: Map<string, string | number>, row: number) {
        this.values = values
        this.rowStart = this.rowEnd = row
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
                const curValue = iset[field]!
                const lastValue = lastValues.get(field)

                if (!madeNewHeader && (curHeader == null || curValue != lastValue)) {
                    const row = curHeader != null ? curHeader.rowEnd : 2
                    curHeader = new GridHeader(new Map(), row)
                    for (var copyIdx = 0; copyIdx < idx; copyIdx ++) {
                        const copyField = FIELDS[copyIdx]
                        curHeader.values.set(copyField, iset[copyField])
                    }
                    headers.push(curHeader)
                    madeNewHeader = true
                }

                curHeader!.values.set(field, curValue)
                lastValues.set(field, curValue)
            }
            curHeader!.rowEnd ++
        }

        this.headers = headers
        return this.headers
    }
}

export { GridHeaders }
