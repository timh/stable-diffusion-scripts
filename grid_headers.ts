import { GImageSet, ColumnHeader, FIELDS, createElement, sort } from "./types.js"
import { GImageGrid } from "./grid.js"

class GridHeaders {
    grid: GImageGrid
    headers: ColumnHeader[]

    constructor(grid: GImageGrid) {
        this.grid = grid
        this.update()
    }

    update(): ColumnHeader[] {
        var lastHeaders = new Map<string, ColumnHeader>() // current header by field
        var headers = new Array<ColumnHeader>()

        // walk through image sets in order, building the columns out
        for (const [isetIdx, isetKey] of this.grid.imageSetKeys.entries()) {
            const iset = this.grid.imageSets.get(isetKey)!

            for (const [idx, field] of FIELDS.entries()) {
                if (field == "modelStr") {
                    continue;
                }
                var header = lastHeaders.get(field)
                if (header == null || header?.value != iset![field]) {
                    const column = (header != null) ? header.columnEnd : 2
                    header = new ColumnHeader(idx + 1, field, iset![field], column)
                    lastHeaders.set(field, header)
                    headers.push(header)
                }
                header.columnEnd ++
            }
        }

        // now walk through the headers again. add classes to each of the headers such that it 
        // correctly nests within the appropriate other headers.
        var curFieldValues = new Map<String, any>()
        var curFieldColumnEnds = new Map<String, number>()
        var headersToUpdate = new Array<ColumnHeader>()
        var curColumn = 2
        for (var i = 0; i <= headers.length; i ++) {
            var header = i < headers.length ? headers[i] : undefined;
            if (header == null || header.columnStart > curColumn) {
                while (headersToUpdate.length > 0) {
                    const toUpdate = headersToUpdate.pop()!
                    var classes = new Array<String>()
                    for (const field of FIELDS) {
                        // do not add a class to toUpdate if its end column is higher
                        // than the driving header (header)
                        if (toUpdate.columnEnd > curFieldColumnEnds.get(field)!) {
                            continue
                        }
                        var value = curFieldValues.get(field)!
                        var valueIndex = this.grid.fieldValueIndex.get(field)!.get(value)!
                        classes.push(`${field}_${valueIndex}`)
                    }
                    toUpdate.classes = classes.join(" ")
                }
            }
            if (header != null) {
                // keep track of current value for each field, and the column that
                // that value ends on. build up a list of headers that need to be updated.
                curFieldValues.set(header.field, header.value)
                curFieldColumnEnds.set(header.field, header.columnEnd)
                curColumn = header.columnStart
                headersToUpdate.push(header)
            }
        }
        this.headers = headers
        return this.headers
    }
}

export { GridHeaders }
