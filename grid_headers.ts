import { GImageSet, FIELDS, createElement, sort } from "./types.js"
import { GImageGrid } from "./grid.js"

class GridHeader {
    rowStart: number = 1
    rowEnd: number = 1
    values: Map<string, string | number>
    classes: string = ""

    constructor(values: Map<string, string | number>, row: number) {
        this.values = values
        this.rowStart = this.rowEnd = row
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


        this.headers = headers
        return this.headers
    }
}

export { GridHeaders }
